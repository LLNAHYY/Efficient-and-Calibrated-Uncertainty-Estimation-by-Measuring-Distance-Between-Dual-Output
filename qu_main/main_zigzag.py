import argparse
import time
from pathlib import Path
import numpy as np
import sys
import os

# Study 폴더를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torchmetrics.functional.classification import multiclass_auroc, multiclass_precision, multiclass_recall, multiclass_accuracy, multiclass_f1_score
from uqmodule.utils import seed_everything,save_text,CustomEarlyStopping
from data.dataset import CIFAR10DataModule, CIFAR100DataModule, MNISTDataModule
from uqmodule.zigzag import ZigZag, zigzag_inference, zigzag_ood_inference
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import pickle

#parser = argparse.ArgumentParser()
#parser.add_argument("--x",type=float,default=1.0,help="What is the first number?")
#parser.add_argument("--y",type=float,default=1.0,help="What is the second number?")
#parser.add_argument("--operation",type=str,default="add",help="What operation? Can choose add, sub, mul, or div")
#args = parser.parse_args()





def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_classes',          type=int,   default=10)
    parser.add_argument('--dataset',     type=str,   default="cifar10")
    parser.add_argument('--epoch',          type=int,   default=100)
    parser.add_argument('--model_name',          type=str,   default="resnet34")
    parser.add_argument('--patience',          type=int,   default=10)
    parser.add_argument('--seeds', type=lambda s: [int(item) for item in s.split(',')], default=[0,93,7777])
    args    = parser.parse_args()

    #num_classes = 10  
    #dataset = "cifar10"  
    #seeds =  [0]
    train_true = True
    #epoch = 1
    qu_method = 'zigzag'

    print(f"method:{qu_method}  data:{args.dataset}")

    if train_true: 
        for seed in args.seeds:
            seed_everything(seed)

            save_path = Path(f"./study/result/{args.dataset}/{qu_method}/{args.model_name}")
            save_path.mkdir(parents=True, exist_ok=True) 

            save_seed_path = Path(save_path, f"seed_{seed}")
            save_seed_path.mkdir(parents=True, exist_ok=True)

            if args.dataset == "cifar10" :
                data_module = CIFAR10DataModule()

            elif args.dataset == "cifar100" :
                data_module = CIFAR100DataModule()
            
            elif args.dataset == "mnist":
                 data_module = MNISTDataModule()

            data_module.prepare_data()
            data_module.setup()
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            test_loader = data_module.test_dataloader()
            ood_loader = data_module.ood_dataloader()
            

            model = ZigZag(args.num_classes, args.model_name)


            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience= args.patience,
                verbose=True,
                mode='min'
            )

            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath= save_seed_path,
                filename="best-{epoch:02d}",
                save_top_k=1,
                mode='min'
            )

            logger = TensorBoardLogger(save_dir=save_seed_path, name="resnet18_zigzag")

            trainer = Trainer(
                default_root_dir=save_seed_path,
                max_epochs=args.epoch,
                logger=logger,
                callbacks=[early_stopping, checkpoint_callback],
                accelerator="gpu", devices=1,
            )
            
            trainer.fit(model, train_loader, val_loader)
            torch.save(model.model.state_dict(), os.path.join(save_seed_path,f'{qu_method}_{seed}_weights.pth'))

            acc, results = zigzag_inference(model, test_loader)

            print(f"{qu_method}_{seed} 정확도: {acc * 100:.2f}%")

            file_name = f"{args.dataset}_test_{seed}.pkl"
            file_path = save_seed_path / file_name

            with open(file_path, 'wb') as f:
                pickle.dump(results, f)

            ood_results = zigzag_ood_inference(model, ood_loader)

            file_ood = f"{args.dataset}_ood_{seed}.pkl"
            file_ood_path = save_seed_path / file_ood

            with open(file_ood_path, 'wb') as f:
                pickle.dump(ood_results, f)


if __name__ == '__main__':
    main()









