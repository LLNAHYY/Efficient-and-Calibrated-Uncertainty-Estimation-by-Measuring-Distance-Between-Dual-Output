import os
import json
import random
import numpy as np
import yaml

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchmetrics import MetricCollection, Accuracy, AUROC, Precision, Recall
from torchmetrics import F1Score as F1

def seed_everything(seed=42):
    pl.seed_everything(seed) # ADDED
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_text(acc , sample_len, total_inference_time, file_path=None):
    with open(file_path, 'w') as f:
        f.write(f"acc: {acc}\n")
        f.write(f"샘플의 수: {sample_len}개\n")
        f.write(f"총 추론 시간: {total_inference_time:.4f} seconds\n")

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor='val_loss', patience=3, mode='min', min_delta=0.0, min_epochs=50, **kwargs):
        super().__init__(monitor=monitor, patience=patience, mode=mode, min_delta=min_delta, **kwargs)
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer, pl_module):
        # EarlyStopping 적용 전에 최소 에포크 확인
        if trainer.current_epoch >= self.min_epochs:
            super().on_validation_end(trainer, pl_module)
        else:
            # EarlyStopping 상태를 초기화하여 영향을 받지 않도록 함
            self.wait_count = 0
            self.stopped_epoch = 0
            # best_score가 None이면 기본값 설정
            if self.best_score is None:
                # 모니터링 기준에 따라 초기화 (mode에 따라 최댓값 또는 최솟값 설정)
                if self.mode == 'min':
                    self.best_score = float('inf')
                elif self.mode == 'max':
                    self.best_score = float('-inf')
