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
from torchvision.models import resnet18, resnet34
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score



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


class resnet(LightningModule):
    def __init__(self, class_num = 10, model_name = 'resnet18'):
        super(resnet, self).__init__()
        if model_name == "resnet18":
            self.model = resnet18(pretrained=True)
        elif model_name == "resnet34":
            self.model = resnet34(pretrained=True)
        elif model_name == "mnist":
            self.model = MNISTMODEL(class_num) 
                   
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, class_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss'  # 검증 손실(val_loss)을 기준으로 학습률을 조정
        }
        return [optimizer], [scheduler]
        
def inference(model, test_loader):
    print("inference")
    model.eval()  
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
            

            output = model(images)
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob,dim=1) 
            uncertainty = calculate_entropy(prob)  # [batch_size] 분산 기반 불확실성 측정 #엔트로피~ 
            
            # Monte Carlo Dropout을 통한 여러 번의 예측
            

            

            outputs.extend(output.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label)  # Convert label to numpy before extending
            severity_levels.extend(severity_level)

    end_time = time.time()
    inference_time  = end_time - start_time
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
    print("ood_inference")
    model.eval()  
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
            

            output = model(images)
            prob = F.softmax(output, dim=1)
            pred = torch.argmax(prob,dim=1) 
            uncertainty = calculate_entropy(prob)  # [batch_size] 분산 기반 불확실성 측정 #엔트로피~ 
            
            # Monte Carlo Dropout을 통한 여러 번의 예측
        


            outputs.extend(output.detach().cpu().numpy())
            probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label)  # Convert label to numpy before extending
            severity_levels.extend(severity_level)

    end_time = time.time()
    inference_time  = end_time - start_time
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
    
    return  results

def calculate_entropy(probabilities):
    """
    엔트로피를 계산하여 불확실성을 측정하는 함수
    
    Args:
        probabilities (torch.Tensor): 각 클래스에 대한 확률 분포, 크기: [batch_size, num_classes]
        
    Returns:
        torch.Tensor: 각 샘플의 엔트로피, 크기: [batch_size]
    """
    # 작은 값을 더해 로그 계산 시 0이 되지 않도록 함
    epsilon = 1e-12
    probabilities = torch.clamp(probabilities, epsilon, 1. - epsilon)
    
    # 엔트로피 계산
    entropy = -torch.sum(probabilities * torch.log(probabilities), dim=1)
    
    return entropy