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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x
    

class CNN_DUQ(Model):
    def __init__(
        self,
        num_classes,
        embedding_size = 256,
        learnable_length_scale = False,
        length_scale = 0.1,
        gamma = 0.999,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def update_embeddings(self, x, y):
        z = self.last_layer(self.compute_features(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()

        return distances

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = self.output_layer(z)

        return y_pred

class ResNet_DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale = 0.1,
        gamma = 0.2,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.feature_extractor(x)
        y_pred = self.rbf(z)

        return y_pred


class DUQ(LightningModule):
    def __init__(self, class_num = 10, model_name = 'resnet18', l_gradient_penalty = 0.01):
        super(DUQ, self).__init__()
        if model_name == "resnet18":
            feature_extractor  = resnet18(pretrained=True)
            feature_extractor.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            feature_extractor.maxpool = torch.nn.Identity()
            feature_extractor.fc = torch.nn.Identity()
            
            self.model = ResNet_DUQ(feature_extractor,class_num,512,512)

        elif model_name == "resnet34":
            feature_extractor  = resnet34(pretrained=True)
            feature_extractor.conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            feature_extractor.maxpool = torch.nn.Identity()
            feature_extractor.fc = torch.nn.Identity()
            
            self.model = ResNet_DUQ(feature_extractor,class_num,512,512)

        elif model_name == "mnist":
            self.model = CNN_DUQ(class_num) 
                   

        self.criterion = nn.CrossEntropyLoss()
        self.class_num = class_num
        self.l_gradient_penalty = l_gradient_penalty

    def forward(self, x):
        return self.model(x)
    
    def calc_gradient_penalty(self, x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)
        grad_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        return gradient_penalty


    def training_step(self, batch, batch_idx):
        images, labels = batch
        images.requires_grad_(True)
        outputs = self(images)
        y_one_hot = F.one_hot(labels, num_classes=self.class_num).float()

        # Calculate loss
        loss = F.binary_cross_entropy(outputs, y_one_hot, reduction="mean")

        # Apply gradient penalty if specified
        if self.l_gradient_penalty > 0:
            gp = self.calc_gradient_penalty(images, outputs)
            loss += self.l_gradient_penalty * gp

        # Log training loss
        self.log("train_loss", loss, on_epoch=True)

        images.requires_grad_(False)

        with torch.no_grad():
            self.model.eval()  # Switch to evaluation mode
            self.model.update_embeddings(images, y_one_hot)
            self.model.train()  # Return to training mode

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        y_one_hot = F.one_hot(labels, num_classes=self.class_num).float()

        # Calculate loss
        loss = F.binary_cross_entropy(outputs, y_one_hot, reduction="mean")
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
            kernel_distance, pred = output.max(1)  # [batch_size] 분산 기반 불확실성 측정 #엔트로피~ 
            epsilon = 1e-8
            distance1  =  1 / (kernel_distance + epsilon)
            
            # Monte Carlo Dropout을 통한 여러 번의 예측
            #outputs.extend(output.detach().cpu().numpy())
            #probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(distance1.detach().cpu().numpy())
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
            #prob = F.softmax(output, dim=1)
            #pred = torch.argmax(prob,dim=1) 
            kernel_distance, pred = output.max(1)  # [batch_size] 분산 기반 불확실성 측정 #엔트로피~ 
            epsilon = 1e-8
            distance1  = 1 / (kernel_distance + epsilon)            
            # Monte Carlo Dropout을 통한 여러 번의 예측
            #outputs.extend(output.detach().cpu().numpy())
            #probs.extend(prob.detach().cpu().numpy())
            preds.extend(pred.cpu().numpy())
            uncertainties.extend(kernel_distance.detach().cpu().numpy())
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