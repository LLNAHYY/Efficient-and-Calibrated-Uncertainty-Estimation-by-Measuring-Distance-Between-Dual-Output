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
from einops import repeat

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision.models import resnet18
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ZigZag(LightningModule):
    def __init__(self, class_num=10, model_name='resnet18'):
        super(ZigZag, self).__init__()

        # 모델 생성
        self.model = self._create_model(class_num,model_name)
        self.criterion = nn.CrossEntropyLoss()
        self.input_linear = False
        self.blank_const = -10

    def _create_model(self, class_num,model_name):
        if model_name == 'resnet18':
            model = resnet18(pretrained=True)
            model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #num_ftrs = model.fc.in_features
            model.fc = nn.Linear(512, class_num)
        elif model_name == 'resnet34':
            model = resnet34(pretrained=True)
            model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #num_ftrs = model.fc.in_features
            model.fc = nn.Linear(512, class_num)
        
        
        return model

    def forward(self, x, y = None, training = False):
        if y is None:
            if self.input_linear:
                x_in = torch.concat(
                    [x, self.blank_const * torch.ones([x.shape[0], 1])], dim=1
                )
            else:
                batch_size, _, height, width = x.shape
                ones_tensor = torch.ones(
                    [batch_size, 1, height, width], device=x.device, dtype=x.dtype
                )
                x_in = torch.cat([x, self.blank_const * ones_tensor], dim=1)
        else:
            if y.dim() == 1:
                y = y.unsqueeze(-1)
            if self.input_linear:
                # classification labels are just 1D
                x_in = torch.concat([x, torch.atleast_2d(y)], dim=1)
            else:
                batch_size, _, height, width = x.shape
                channel_y = torch.atleast_2d(y).shape[-1]
                ones_tensor = torch.ones(
                    [batch_size, channel_y, height, width],
                    device=x.device,
                    dtype=x.dtype,
                )
                if training:
                    inputs_1 = torch.cat([x, self.blank_const * ones_tensor], dim=1)
                    # The second input with actual targets, the second term in Eq. 1
                    t_inputs = y.reshape(-1, 1, 1, 1) * ones_tensor
                    inputs_2 = torch.cat([x, t_inputs], dim=1)

                    p = 0.5
                    mask = (
                        (torch.empty(inputs_1.shape[0], 1, 1, 1).uniform_(0, 1) > p)
                        .float()
                        .to(x.device)
                    )
                    x_in = inputs_1 * mask + inputs_2 * (1 - mask)
                else:
                    x_in = torch.cat([x, y.reshape(-1, 1, 1, 1) * ones_tensor], dim=1)

        return self.model(x_in)


    def training_step(self, batch, batch_idx):
        X,y = batch
        x_in = X
        y_in = y
        y_target = y

        out = self.forward(x_in, y_in, training=True)
        loss = self.criterion(out, y_target)

        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        x_in = X
        y_in = y
        y_target = y

        out = self.forward(x_in, y_in, training=False)
        loss = self.criterion(out, y_target)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss'  
        }
        return [optimizer], [scheduler]

    #def ensemble_predict(self, x):
        # 각 모델의 예측을 수집하여 평균 및 불확실성 계산
    #    outputs = torch.stack([F.softmax(model(x), dim=1) for model in self.models])
    #    mean_outputs = outputs.mean(dim=0)  # 평균 예측
    #    uncertainty = outputs.var(dim=0).mean(dim=1)  # 불확실성 계산 (분산 기반)

    #    return mean_outputs, uncertainty


from sklearn.metrics import accuracy_score

def zigzag_inference(model, test_loader):

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
            label = label.to('cuda')

            
            Y_1 = model(images, training=False)
            Y_1_softmax = torch.softmax(Y_1, dim=1)
            Y_1_labels = torch.argmax(Y_1_softmax, dim=1)
            Y_2 = model(images, Y_1_labels, training=False)
            Y_2_softmax = torch.softmax(Y_2, dim=1)

            
            uncertainty = torch.abs(Y_1_softmax - Y_2_softmax).mean(dim=1)

            outputs.extend(Y_1.detach().cpu().numpy())
            probs.extend(Y_1_softmax.detach().cpu().numpy())
            preds.extend(Y_1_labels.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
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


def zigzag_ood_inference(model, test_loader):
    print("OOD INFERENCE START")
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
            images = images.to('cuda')
            label = label.to('cuda')
            
            start_time = time.time()

            Y_1 = model(images, training=False)
            Y_1_softmax = torch.softmax(Y_1, dim=1)
            Y_1_labels = torch.argmax(Y_1_softmax, dim=1)
            Y_2 = model(images, Y_1_labels, training=False)
            Y_2_softmax = torch.softmax(Y_2, dim=1)
            uncertainty = torch.abs(Y_1_softmax - Y_2_softmax).mean(dim=1)

            outputs.extend(Y_1.detach().cpu().numpy())
            probs.extend(Y_1_softmax.detach().cpu().numpy())
            preds.extend(Y_1_labels.cpu().numpy())
            uncertainties.extend(uncertainty.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
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
