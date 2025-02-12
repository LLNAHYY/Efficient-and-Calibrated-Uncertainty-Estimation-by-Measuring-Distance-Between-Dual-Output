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





class MNISTMODEL(nn.Module):
    def __init__(self, num_classes = 10):
        super(MNISTMODEL, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ELU(),  # 활성화 함수 추가
            Masksembles2D(32, 4, 2),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 두 번째 합성곱 + 활성화 + 풀링 레이어
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ELU(),  # 활성화 함수 추가
            Masksembles2D(64, 4, 2),
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


class ResNet18MaskedEnsemble(LightningModule):
    def __init__(self, class_num=10, model_name = "resnet18", ensemble_size=5, pretrained=True, mask_channels=64, mask_n=4, mask_scale=2.0):
        super(ResNet18MaskedEnsemble, self).__init__()
        self.ensemble_size = ensemble_size

        # 여러 ResNet18 모델을 모듈 리스트로 생성
        self.model = nn.ModuleList([
            self._create_model(class_num, model_name, pretrained, mask_channels, mask_n, mask_scale)
            for _ in range(ensemble_size)
        ])
        self.criterion = nn.CrossEntropyLoss()

    def _create_model(self, class_num, model_name, pretrained, mask_channels, mask_n, mask_scale):

        if model_name == "resnet18":
            model = resnet18(pretrained=pretrained)
            model.layer1 = nn.Sequential(
                model.layer1,
                Masksembles2D(mask_channels, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer2 = nn.Sequential(
                model.layer2,
                Masksembles2D(mask_channels * 2, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer3 = nn.Sequential(
                model.layer3,
                Masksembles2D(mask_channels * 4, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer4 = nn.Sequential(
                model.layer4,
                Masksembles2D(mask_channels * 8, mask_n, mask_scale)  # Masksembles2D 추가
            )

            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                Masksembles1D(num_ftrs, mask_n, mask_scale),  # Masksembles1D 추가
                nn.Linear(num_ftrs, class_num)
            )
        elif model_name == "resnet34":
            model = resnet34(pretrained=pretrained)
            model.layer1 = nn.Sequential(
                model.layer1,
                Masksembles2D(mask_channels, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer2 = nn.Sequential(
                model.layer2,
                Masksembles2D(mask_channels * 2, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer3 = nn.Sequential(
                model.layer3,
                Masksembles2D(mask_channels * 4, mask_n, mask_scale)  # Masksembles2D 추가
            )
            model.layer4 = nn.Sequential(
                model.layer4,
                Masksembles2D(mask_channels * 8, mask_n, mask_scale)  # Masksembles2D 추가
            )

            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                Masksembles1D(num_ftrs, mask_n, mask_scale),  # Masksembles1D 추가
                nn.Linear(num_ftrs, class_num)
            )

        elif model_name == "mnist":
            model = MNISTMODEL(class_num)

        return model

    def forward(self, x):
        # Ensemble 모델들의 예측을 리스트로 반환
        outputs = [model(x) for model in self.model]
        return outputs #[batch, class] 가 emsemble_size 만큼 존재 

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
            'monitor': 'val_loss'  # 검증 손실(val_loss)을 기준으로 학습률을 조정
        }
        return [optimizer], [scheduler]

    #def ensemble_predict(self, x):
        # 각 모델의 예측을 수집하여 평균 및 불확실성 계산
    #    outputs = torch.stack([F.softmax(model(x), dim=1) for model in self.models])
    #    mean_outputs = outputs.mean(dim=0)  # 평균 예측
    #    uncertainty = outputs.var(dim=0).mean(dim=1)  # 불확실성 계산 (분산 기반)

    #    return mean_outputs, uncertainty
#--------------------------------------------https://github.com/nikitadurasov/masksembles/blob/main/masksembles/torch.py------------------------------------------------------------------------------
import numpy as np

import torch
from torch import nn



class Masksembles2D(nn.Module):
    """
    :class:`Masksembles2D` is high-level class that implements Masksembles approach
    for 2-dimensional inputs (similar to :class:`torch.nn.Dropout2d`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C, H, W)
        * Output: (N, C, H, W) (same shape as input)

    Examples:

    >>> m = Masksembles2D(16, 4, 2.0)
    >>> input = torch.ones([4, 16, 28, 28])
    >>> output = m(input)

    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):
        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        masks = generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).double()

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2, 3, 4])
        x = x * self.masks.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0).float()
    
class Masksembles1D(nn.Module):
    """
    :class:`Masksembles1D` is high-level class that implements Masksembles approach
    for 1-dimensional inputs (similar to :class:`torch.nn.Dropout`).

    :param channels: int, number of channels used in masks.
    :param n: int, number of masks
    :param scale: float, scale parameter similar to *S* in [1]. Larger values decrease \
        subnetworks correlations but at the same time decrease capacity of every individual model.

    Shape:
        * Input: (N, C)
        * Output: (N, C) (same shape as input)

    Examples:

    >>> m = Masksembles1D(16, 4, 2.0)
    >>> input = torch.ones([4, 16])
    >>> output = m(input)


    References:

    [1] `Masksembles for Uncertainty Estimation`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua

    """

    def __init__(self, channels: int, n: int, scale: float):

        super().__init__()

        self.channels = channels
        self.n = n
        self.scale = scale

        masks = generation_wrapper(channels, n, scale)
        masks = torch.from_numpy(masks)
        self.masks = torch.nn.Parameter(masks, requires_grad=False).double()

    def forward(self, inputs):
        batch = inputs.shape[0]
        x = torch.split(inputs.unsqueeze(1), batch // self.n, dim=0)
        x = torch.cat(x, dim=1).permute([1, 0, 2])
        x = x * self.masks.unsqueeze(1)
        x = torch.cat(torch.split(x, 1, dim=0), dim=1)
        return x.squeeze(0).float()

def generate_masks_(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.

    Results of this function are stochastic, that is, calls with the same sets
    of arguments might generate outputs of different shapes. Check generate_masks
    and generation_wrapper function for more deterministic behaviour.

    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    total_positions = int(m * s)
    masks = []

    for _ in range(n):
        new_vector = np.zeros([total_positions])
        idx = np.random.choice(range(total_positions), m, replace=False)
        new_vector[idx] = 1
        masks.append(new_vector)

    masks = np.array(masks)
    # drop useless positions
    masks = masks[:, ~np.all(masks == 0, axis=0)]
    return masks


def generate_masks(m: int, n: int, s: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by n, m, s params.

    Resulting masks are required to have fixed features size as it's described in [1].
    Since process of masks generation is stochastic therefore function evaluates
    generate_masks_ multiple times till expected size is acquired.

    :param m: int, number of ones in each mask
    :param n: int, number of masks in the set
    :param s: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors

    References

    [1] `Masksembles for Uncertainty Estimation: Supplementary Material`,
    Nikita Durasov, Timur Bagautdinov, Pierre Baque, Pascal Fua
    """

    masks = generate_masks_(m, n, s)
    # hardcoded formula for expected size, check reference
    expected_size = int(m * s * (1 - (1 - 1 / s) ** n))
    while masks.shape[1] != expected_size:
        masks = generate_masks_(m, n, s)
    return masks


def generation_wrapper(c: int, n: int, scale: float) -> np.ndarray:
    """Generates set of binary masks with properties defined by c, n, scale params.

     Allows to generate masks sets with predefined features number c. Particularly
     convenient to use in torch-like layers where one need to define shapes inputs
     tensors beforehand.

    :param c: int, number of channels in generated masks
    :param n: int, number of masks in the set
    :param scale: float, scale param controls overlap of generated masks
    :return: np.ndarray, matrix of binary vectors
    """

    if c < 10:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"number of channels is less then 10. Current value is (channels={c}). "
                         "Please increase number of features in your layer or remove this "
                         "particular instance of Masksembles from your architecture.")

    if scale > 6.:
        raise ValueError("Masksembles approach couldn't be used in such setups where "
                         f"scale parameter is larger then 6. Current value is (scale={scale}).")
    
    # inverse formula for number of active features in masks
    active_features = int(int(c) / (scale * (1 - (1 - 1 / scale) ** n)))

    # Fix the last part by using binary search 
    max_iter = 1000

    min = np.max([scale * 0.8, 1.0])
    max = scale * 1.2

    for _ in range(max_iter):
        mid = (min + max) / 2
        masks = generate_masks(active_features, n, mid)
        if masks.shape[-1] == c:
            break
        elif masks.shape[-1] > c:
            max = mid
        else:
            min = mid

    if masks.shape[-1] != c:
        raise ValueError("generation_wrapper function failed to generate masks with "
                         "requested number of features. Please try to change scale parameter")

    return masks
#--------------------------------------------https://github.com/nikitadurasov/masksembles/blob/main/masksembles/torch.py------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score



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
            images = images.to(torch.float).to('cuda')
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
