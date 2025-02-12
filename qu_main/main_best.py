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
from uqmodule.utils import seed_everything,CustomEarlyStopping
from data.dataset import CIFAR10DataModule, CIFAR100DataModule, MNISTDataModule
from study.uqmodule.EUDD import inference , ood_inference,pretrained
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, LightningModule
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import pickle






def main():

    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    # 입력받을 인자값 설정 (default 값 설정가능)
    parser.add_argument('--num_classes',          type=int,   default=10)
    parser.add_argument('--dataset',     type=str,   default="cifar10")
    parser.add_argument('--epoch',          type=int,   default=150)
    parser.add_argument('--model_name',          type=str,   default="resnet34")
    parser.add_argument('--patience',          type=int,   default=10)
    parser.add_argument('--seeds', type=lambda s: [int(item) for item in s.split(',')], default=[0,93,7777])
    args    = parser.parse_args()

    train_true = True
    qu_method = 'my'
    print(f"method:{qu_method}  data:{args.dataset}")

    if train_true: 
        #for p in lists:
            for seed in args.seeds:
                seed_everything(seed)

                save_path = Path(f"./study/result/{args.dataset}/{qu_method}/{args.model_name}/none")
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
                
                pre_model = pretrained(args.num_classes, args.model_name,  30)


                for param in pre_model.model.fc_mu.parameters():
                    param.requires_grad = False


                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience = args.patience,
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

                trainer.fit(pre_model, train_loader, val_loader)
                #torch.save(pre_model.model.state_dict(), os.path.join(save_seed_path,f'{qu_method}_{seed}_weights.pth'))
                optimal_temperature = pre_model.temperature.item()
                upper_95_percentile = pre_model.upper_95_percentile.item()

                print(optimal_temperature)
                print(upper_95_percentile)


            

           # pre_model.model.load_state_dict(torch.load(os.path.join(save_seed_path,f'{qu_method}_{seed}_weights.pth')))


                acc, results = inference(pre_model, test_loader, upper_95_percentile, optimal_temperature)

                print(f"{qu_method}_{seed} 정확도1: {acc * 100:.2f}%")


                file_name = f"{args.dataset}_test_{seed}.pkl"
                file_path = save_seed_path / file_name

                with open(file_path, 'wb') as f:
                    pickle.dump(results, f)

                

                ood_results = ood_inference(pre_model, ood_loader, upper_95_percentile, optimal_temperature)
                
                file_ood = f"{args.dataset}_ood_{seed}.pkl"
                file_ood_path = save_seed_path / file_ood

                with open(file_ood_path, 'wb') as f:
                    pickle.dump(ood_results, f)


            print("Results saved")


if __name__ == '__main__':
    main()









