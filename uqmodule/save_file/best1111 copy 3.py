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
            nn.ELU(), 
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(), 
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
    def __init__(self, num_classes, model_name , latent_dim=128, p = 30):
        super(ResNetVAEWithFeature, self).__init__()
        
        #0 ResNet-18을 불러오고 마지막 fully connected 레이어 제거
        if model_name == "resnet18":
            model = resnet18(pretrained=True)
        elif model_name == "resnet34":
            model = resnet34(pretrained=True)
        elif model_name == "mnist":
            model = MNISTMODEL(num_classes)

        self.p = p

        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])  # 마지막 FC 레이어 제거

        self.fc_feature = nn.Linear(model.fc.in_features, latent_dim)
        self.fc_mu = nn.Linear(model.fc.in_features, latent_dim)
        self.fc_logvar = nn.Linear(model.fc.in_features, latent_dim)


        self.fc_class1 = nn.Linear(latent_dim, num_classes)
        self.fc_class2 = nn.Linear(latent_dim, num_classes)

        self.dropout = nn.Dropout(0.3)

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
        z_mean = z_samples.sum(dim=0)
        
        return z_mean

    def forward(self, x):
        h = self.feature_extractor(x)
        h = torch.flatten(h, 1)  

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        #mu = F.relu(mu)
        #mu = self.dropout(mu) 



        #z1 = F.relu(self.fc1(mu))
        #z1 = self.dropout(z1) 
        f = self.fc_feature(h)

        z1 = self.sample_z(mu, logvar , 1) #self.p)
        #z2 = self.sample_z(mu, logvar, num_samples = 1)

        #z = mu + torch.exp(0.5 * logvar) * torch.rand_like(logvar)
        #epsilon = torch.randn_like(mu, device=mu.device)

        class_output1 = self.fc_class1(f)
        class_output2 = self.fc_class2(z1)

        return class_output1, class_output2

class pretrained(LightningModule):
    def __init__(self, class_num=10, model_name = 'resnet18', s = 30):
        super(pretrained, self).__init__()

        # 모델 생성
        self.model = ResNetVAEWithFeature( num_classes = class_num, model_name = model_name, p = s)
        self.criterion1 = FocalLoss () #nn.CrossEntropyLoss()
        self.criterion2 = nn.CrossEntropyLoss()
        self.upper_95_percentile = None
        self.temperature  = torch.nn.Parameter(torch.ones(1) * 1.0)

        if model_name == "resnet18":
            self.p = 40
        elif model_name == "resnet34":
            self.p = 50
        elif model_name == "mnist":
            self.p = 50


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        class_output1, class_output2 = self.model(images)
        loss1 = self.criterion1(class_output1,labels)
        loss2 = self.criterion2(class_output1,labels)
        loss3 = self.criterion1(class_output2,labels)
        
        loss = loss1 + 0.01 * loss3
        #oss = 0.5 *loss1 + 0.5 * loss2 + 0.01 * loss3

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        class_output1, class_output2 = self.model(images)
        loss1 = self.criterion1(class_output1,labels)
        loss2 = self.criterion2(class_output1,labels)
        #loss = 0.5 *loss1 + 0.5 * loss2
        loss = loss1 
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5),
            'monitor': 'val_loss'  
        }
        return [optimizer], [scheduler]
    
    def temperature_scaled_logits(self, logits, temperature):
        temperature = torch.exp(temperature) + 1

        return logits / temperature
    
    def mixup_data(self, images, labels, alpha=0.3):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)
        
        mixed_images = lam * images + (1 - lam) * images[index, :]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam

    
    def on_train_end(self):
        self.model.eval()

        uncertaintys = []

        val_loader = self.trainer.val_dataloaders

        # uncertainty 구하기 
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(self.device) 
                labels = labels.to(self.device) 
                
                # 모델 예측
                class_output1, class_output2 = self.model(images)                
                epsilon = 1e-8
                distance1  = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1)) 
                distance1  = 1 / (distance1 + epsilon) 
                uncertaintys.extend(distance1.detach().cpu().numpy())
        
        self.upper_95_percentile = np.percentile(uncertaintys, 60)
        upper_50_threshold = np.percentile(uncertaintys, self.p)
        temperature_optimizer = torch.optim.Adam([self.temperature], lr=0.01)
        
        # val_loader 반복문에서 불확실성이 50% 이하인 데이터만 최적화에 사용
        for batch in val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 모델 예측 및 불확실성 계산
            with torch.no_grad():
                logits, _ = self.model(images)  # 단일 모델 예측
                class_output1, class_output2 = logits, _  # 필요한 경우 업데이트 필요
                epsilon = 1e-8
                distance1 = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1)) #여기선 entropy 안곱했을 때 효과가 제일 좋음 주의
                distance1 = 1 / (distance1 + epsilon) 

            # 상위 50% 이하의 불확실성을 가진 데이터 필터링
            mask = distance1 <= upper_50_threshold
            if mask.sum() == 0:
                continue  # 상위 50% 이하 데이터가 없으면 건너뜁니다.

            # 선택된 데이터만 온도 최적화에 사용
            selected_logits = logits[mask]
            selected_labels = labels[mask]

            # 온도 스케일링 적용 후 손실 계산 및 최적화
            temperature_optimizer.zero_grad()
            scaled_logits = self.temperature_scaled_logits(selected_logits, self.temperature)
            loss = self.criterion2(scaled_logits, selected_labels)
            loss.backward()
            temperature_optimizer.step()


def inference(model, test_loader, threshold,Temperautue):

    print("inference")
    model.eval()  
    model = model.to('cuda')
    #threshold = threshold*3

    outputs = []    
    preds = []
    probs = []
    uncertainties = []
    labels = []
    severity_levels = []
    OODs = []

    inference_times = 0
    T = np.exp(Temperautue) + 1
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to(torch.float).to('cuda')
            label = label.to('cuda')
            epsilon = 1e-8
        # 모델 예측
            class_output1, class_output2 = model(images)
            prob = F.softmax(class_output1, dim=1)
            distance1 = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1))
            distance1 = 1 / (distance1 + epsilon) #* calculate_entropy(F.softmax(class_output1, dim=1))
            uncertainty = distance1 * calculate_entropy(prob)
            binary_distance = torch.where(distance1 < threshold, torch.zeros_like(distance1), torch.ones_like(distance1))


            scaled_logits = class_output1/ T
            #scaling_mask = binary_distance == 0  

            #if scaling_mask.any():
            #    scaled_logits[scaling_mask] = class_output1[scaling_mask] / T

            # 최종 확률 계산
            prob1 = F.softmax(scaled_logits, dim=1)
            #_entropy(prob1) #distance1 * calculate_entropy(prob1) #* 100
            preds1 = prob1.argmax(dim=1) 

            OODs.extend(binary_distance.cpu().numpy())
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
                'acc' : acc,
                "OOD" : OODs

                }
    
    return acc ,results

def ood_inference(model, test_loader, threshold,Temperautue):

    print("zigzag_ood_inference")
    model.eval()  # Ensemble 모델을 평가 모드로 전환
    model = model.to('cuda')
    T = np.exp(Temperautue) + 1
    #threshold = threshold*3

    
    outputs = []    
    preds = []
    probs = []
    uncertainties = []
    labels = []
    severity_levels = []
    OODs= []

    inference_times = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, label, severity_level) in enumerate(tqdm(test_loader)):
            images = images.to(torch.float).to('cuda')
            label = label.to('cuda')
            epsilon = 1e-8
            # 모델 예측
            class_output1, class_output2 = model(images)
            prob = F.softmax(class_output1, dim=1)
            distance1 = torch.sqrt(torch.sum((class_output1 - class_output2) ** 2, dim=1))
            distance1 = 1 / (distance1 + epsilon) #* calculate_entropy(F.softmax(class_output1, dim=1))
            uncertainty = distance1 * calculate_entropy(prob)
            binary_distance = torch.where(distance1 < threshold, torch.zeros_like(distance1), torch.ones_like(distance1))

            scaled_logits = class_output1/ T
            #scaling_mask = binary_distance == 0  

            #if scaling_mask.any():
            #    scaled_logits[scaling_mask] = class_output1[scaling_mask] / T

            # 최종 확률 계산
            prob1 = F.softmax(scaled_logits, dim=1)
            #uncertainty =  distance1 * calculate_entropy(prob1) #calculate_entropy(prob1) #distance1 * calculate_entropy(prob1) #* 100
            preds1 = prob1.argmax(dim=1) 

            OODs.extend(binary_distance.cpu().numpy())

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
                "inference_times": inference_time,
                "OOD" : OODs
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