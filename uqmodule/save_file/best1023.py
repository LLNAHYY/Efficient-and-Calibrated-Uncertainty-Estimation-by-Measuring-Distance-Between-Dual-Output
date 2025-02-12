import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
from pytorch_lightning import LightningModule
from torchvision.models import resnet18,resnet34
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        

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


# 모델 정의
class ResNetVAEWithFeature(nn.Module):
    def __init__(self, num_classes, model_name , latent_dim=128, feature_dim=512):
        super(ResNetVAEWithFeature, self).__init__()
        
        #0 ResNet-18을 불러오고 마지막 fully connected 레이어 제거
        if model_name == "resnet18":
            model = resnet18(pretrained=True)
        elif model_name == "resnet34":
            model = resnet34(pretrained=True)
        elif model_name == "mnist":
            model = MNISTMODEL(num_classes)

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # 마지막 FC 레이어 제거

        self.fc_feature = nn.Linear(model.fc.in_features, latent_dim)
        self.fc_mu = nn.Linear(model.fc.in_features, latent_dim)
        self.fc_logvar = nn.Linear(model.fc.in_features, latent_dim)


        self.fc_class1 = nn.Linear(latent_dim, num_classes)
        self.fc_class2 = nn.Linear(latent_dim, num_classes)

        # layer1의 가중치와 편향을 layer2에 복사
        #self.fc_class2.weight.data = self.fc_class1.weight.data.clone()

        #self.fc_class2.bias.data = self.fc_class1.bias.data.clone()
        #self.fc_mu.weight.data = self.fc_feature.weight.data.clone()
        #self.fc_mu.bias.data = self.fc_feature.bias.data.clone()
        #self.fc_logvar.weight.data = self.fc_feature.weight.data.clone()
        #self.fc_logvar.bias.data = self.fc_feature.bias.data.clone()
    def sample_z(self, mu, logvar, num_samples=30):
        # 30번 샘플링 후 평균 계산
        batch_size, latent_dim = mu.size()
        
        # 샘플링을 한 번에 수행
        epsilon = torch.randn((num_samples, batch_size, latent_dim), device=mu.device)
        z_samples = mu.unsqueeze(0) + epsilon * torch.exp(0.5 * logvar).unsqueeze(0)
        
        # 샘플의 평균 계산
        z_mean = z_samples.mean(dim=0)
        
        return z_mean

    def forward(self, x):
        h = self.feature_extractor(x)
        h = torch.flatten(h, 1)  

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        #f = self.fc_feature(h)

        z1 = self.sample_z(mu, logvar)
        #z2 = self.sample_z(mu, logvar, num_samples = 1)

        #z = mu + torch.exp(0.5 * logvar) * torch.rand_like(logvar)
        #epsilon = torch.randn_like(mu, device=mu.device)


        class_output1 = self.fc_class1(z1)
        class_output2 = self.fc_class2(z1)

        return class_output1, class_output2

class pretrained(LightningModule):
    def __init__(self, class_num=10, model_name = 'resnet18'):
        super(pretrained, self).__init__()

        # 모델 생성
        self.model = ResNetVAEWithFeature(class_num, model_name)
        self.criterion1 = FocalLoss () #nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        class_output1, class_output2 = self.model(images)
        loss1 = self.criterion1(class_output1,labels)
        loss2 = self.criterion2(class_output1,labels)
        loss3 = self.criterion2(class_output2,labels)

        loss = 0.5 * loss1 + 0.5 * loss2 - 0.001 * loss3

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        class_output1, class_output2 = self.model(images)
        loss1 = self.criterion1(class_output1,labels)
        loss2 = self.criterion2(class_output1,labels)
        loss3 = self.criterion2(class_output2,labels)

        loss = 0.5 * loss1 + 0.5 * loss2 - 0.001 * loss3
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss'  
        }
        return [optimizer], [scheduler]




def inference(model, test_loader):

    print("inference")
    model.eval()  
    model = model.to('cuda')

    outputs = []    
    preds = []
    probs = []
    uncertainties = []
    labels = []
    severity_levels = []

    inference_times = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to(torch.float).to('cuda')
            label = label.to('cuda')
            epsilon = 1e-8

            class_output1, class_output2  = model(images)
            prob1 = F.softmax(class_output1, dim=1)
            prob2 = F.softmax(class_output2, dim=1)
            distance1  = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1)) 
            distance1  = 100 / (distance1 + epsilon)

            epsilon = 1e-8
            uncertainty =  distance1 * calculate_entropy(prob1) #* 100
            preds1 = prob1.argmax(dim=1) 


            outputs.extend(class_output1.cpu().numpy())
            preds.extend(preds1.cpu().numpy())
            probs.extend(prob1.detach().cpu().numpy())
            labels.extend(label.cpu().numpy())
            severity_levels.extend(severity_level)
            uncertainties.extend(uncertainty.detach().cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time
    inference_times += inference_time
    acc = accuracy_score(preds,labels)

    results = {
                "output" : outputs,
                "prob": probs,
                "pred": preds,
                "label": labels,
                "type": severity_levels,
                "uncertainty": uncertainties,
                "inference_times": inference_time,
                'acc' : acc
                }
    
    return acc ,results

def ood_inference(model, test_loader):

    print("zigzag_ood_inference")
    model.eval()  # Ensemble 모델을 평가 모드로 전환
    model = model.to('cuda')
    
    outputs = []    
    preds = []
    probs = []
    uncertainties = []
    labels = []
    severity_levels = []

    inference_times = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to(torch.float).to('cuda')
            label = label.to('cuda')
            epsilon = 1e-8

            class_output1, class_output2  = model(images)
            prob1 = F.softmax(class_output1, dim=1)
            prob2 = F.softmax(class_output2, dim=1)
            distance1  = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1)) 
            distance1  = 100 / (distance1 + epsilon)

            epsilon = 1e-8
            uncertainty =  distance1  * calculate_entropy(prob1) #* 100
            prob1 = F.softmax(class_output1, dim=1)
            preds1 = prob1.argmax(dim=1) 

            outputs.extend(class_output1.cpu().numpy())
            preds.extend(preds1.cpu().numpy())
            probs.extend(prob1.detach().cpu().numpy())
            labels.extend(label.cpu().numpy())
            severity_levels.extend(severity_level)
            uncertainties.extend(uncertainty.detach().cpu().numpy())

    end_time = time.time()
    inference_time = end_time - start_time
    inference_times += inference_time

    results = {
                "output" : outputs,
                "prob": probs,
                "pred": preds,
                "label": labels,
                "type": severity_levels,
                "uncertainty": uncertainties,
                "inference_times": inference_time
                }
    
    return results

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