import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import resnet18,resnet34
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MNISTMODEL(nn.Module):
    def __init__(self, num_classes = 10):
        super(MNISTMODEL, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ELU(),  # 활성화 함수 추가
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 두 번째 합성곱 + 활성화 + 풀링 레이어
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),  # 활성화 함수 추가
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 5 * 5, out_features=num_classes) 

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x  = self.flatten(x)
        x = self.fc(x)
        return x



class ResNet18Ensemble(LightningModule):
    def __init__(self, class_num=10, model_name = 'resnet18' ,ensemble_size=5, epsilon=0.01):
        super(ResNet18Ensemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.epsilon = epsilon  # 적대적 예제의 크기
        self.model_name = model_name

        # Ensemble을 구성하는 여러 모델을 생성
        self.model = nn.ModuleList([self._create_model(class_num, model_name) for _ in range(ensemble_size)])
        self.criterion = nn.CrossEntropyLoss()

    def _create_model(self, class_num, model_name):

        if model_name == "resnet18":
            model = resnet18(pretrained=True)
        elif model_name == "resnet34":
            model = resnet34(pretrained=True)
        elif model_name == "mnist":
            model = MNISTMODEL(class_num)

        # ResNet18 모델 생성
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_num)
        return model

    def forward(self, x):
        # Ensemble 모델들의 예측을 리스트로 반환
        outputs = [model(x) for model in self.model]
        return outputs  # [batch, class] 가 ensemble_size 만큼 존재 

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # 각 모델에 대해 loss 계산 및 평균
        losses = []
        for sub_model in self.model:
            outputs = sub_model(images)
            loss = self.criterion(outputs, labels)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('train_loss', avg_loss)
        return avg_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        # 각 모델의 예측을 모은 후 평균 loss 계산
        losses = []
        for sub_model in self.model:
            outputs = sub_model(images)
            loss = self.criterion(outputs, labels)
            losses.append(loss)
        avg_loss = torch.stack(losses).mean()
        self.log('val_loss', avg_loss)
        return avg_loss
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]



def inference(model, test_loader):
    """
    model: ResNet18EnsembleModel() 처럼 여러 모델을 포함한 Ensemble 모델
    test_loader: 테스트 데이터 로더
    """
    print("INFERENCE START")
    model.eval()  # Ensemble 모델을 평가 모드로 전환
    model = model.to('cuda')
    
    probs = []
    preds = []
    outputs = []
    uncertainties = []
    labels = []
    severity_levels = []
    
    inference_times = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to(torch.float).to('cuda')

            # 앙상블의 각 모델을 사용하여 예측 수
            # Ensemble 모델에서 바로 예측
            model_output = model(images)  # 모델 내에서 여러 모델의 예측을 수행하고 결합된 결과 반환
            output = [F.softmax(x, dim=1) for x in model_output]  # [num_samples, batch_size, num_classes]
            output = torch.stack(output)  # 리스트를 텐서로 변환 [num_samples, batch_size, num_classes]
            
            prob = output.mean(dim=0)  # [batch_size, num_classes] 평균 예측
            uncertainty = output.var(dim=0).mean(dim=1)  # [batch_size] 분산 기반 불확실성 측정 #엔트로피~ 
            pred = prob.argmax(dim=1)


            
            mean_logit = [x for x in model_output] 
            mean_logit = torch.stack(mean_logit).mean(dim=0)
        

            outputs.extend(mean_logit.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label)
            severity_levels.extend(severity_level)

    end_time = time.time()
    inference_time = end_time - start_time
    inference_times += inference_time 

    acc = accuracy_score(preds,labels)

    results = {
            "output": outputs,
            "prob": probs,
            "pred": preds,
            "uncertainty": uncertainties,
            "label": labels,
            "type": severity_levels,
            "inference_times": inference_time,
            "acc" : acc
        }
            
    return acc, results

def ood_inference(model, test_loader):
    """
    model: ResNet18EnsembleModel() 처럼 여러 모델을 포함한 Ensemble 모델
    test_loader: 테스트 데이터 로더
    """
    print("OOD INFERENCE START")
    model.eval()  # Ensemble 모델을 평가 모드로 전환
    model = model.to('cuda')
    
    probs = []
    preds = []
    outputs = []
    uncertainties = []
    labels = []
    severity_levels = []
    
    inference_times = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to('cuda')
            #avg_logits = torch.zeros(images.size(0), model.model[0].fc.out_features).to(images.device)
            logits_list = []

            # 앙상블의 각 모델을 사용하여 예측 수
            # Ensemble 모델에서 바로 예측
            model_output = model(images)  # 모델 내에서 여러 모델의 예측을 수행하고 결합된 결과 반환
            output = [F.softmax(x, dim=1) for x in model_output]  # [num_samples, batch_size, num_classes]
            output = torch.stack(output)  # 리스트를 텐서로 변환 [num_samples, batch_size, num_classes]
            
            prob = output.mean(dim=0)  # [batch_size, num_classes] 평균 예측
            uncertainty = output.var(dim=0).mean(dim=1)  # [batch_size] 분산 기반 불확실성 측정
            pred = prob.argmax(dim=1)

            
            mean_logit = [x for x in model_output] 
            mean_logit = torch.stack(mean_logit).mean(dim=0)
        

            outputs.extend(mean_logit.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label)
            severity_levels.extend(severity_level)

    end_time = time.time()
    inference_time = end_time - start_time
    inference_times += inference_time 

    results = {
            "output": outputs,
            "prob": probs,
            "pred": preds,
            "uncertainty": uncertainties,
            "label": labels,
            "type": severity_levels,
            "inference_times": inference_time
        }
            
    return results