import sys
sys.path.append('./input/pytorch-image-models-master')
import timm

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn 
import torch.nn.functional as F 
from torch.optim.lr_scheduler import _LRScheduler
from math import sqrt

import utils
from dataset import ShopeeDataset
from preprocess import get_train_transforms
from train_config import Config
from effmodel import ArcMarginProduct, ShopeeScheduler, ShopeeModel

DATA_DIR = './input/shopee-product-matching/train_images'
TRAIN_CSV = './input/shopee-folds/folds-random-10.csv'
MODEL_PATH = './models/'

def run_training():
    
    df = pd.read_csv(TRAIN_CSV)

    labelencoder= LabelEncoder()
    df['label_group'] = labelencoder.fit_transform(df['label_group'])
    
    train = df[df['fold']>1].reset_index(drop=True)
    valid = df[df['fold']==1].reset_index(drop=True)

    trainset = ShopeeDataset(train,
                             DATA_DIR,
                             transform = get_train_transforms(img_size = Config.img_size))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size = Config.batch_size,
        num_workers = Config.num_workers,
        pin_memory = True,
        shuffle = True,
        drop_last = True
    )

    valset = ShopeeDataset(valid,
                             DATA_DIR,
                             transform = get_train_transforms(img_size = Config.img_size))

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size = Config.batch_size,
        num_workers = Config.num_workers,
        pin_memory = True,
        shuffle = False,
        drop_last = False
    )
    
    model = ShopeeModel()
    model.to(Config.device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = Config.scheduler_params['lr_start'])
    scheduler = ShopeeScheduler(optimizer, **Config.scheduler_params)

    for epoch in range(Config.epochs):
        avg_loss_train = utils.train_fn(model, trainloader, optimizer, scheduler, epoch, Config.device)
        
        # avg_loss_test = utils.eval_fn(model, valloader, epoch, Config.device)
        # print(avg_loss_test)
        
        torch.save(model.state_dict(), MODEL_PATH + f'effb3_lr1e5sq2_decay8e-1_warmup5_batch8_epoch18_{epoch}.pt'.format(Config.model_name))
        

if __name__ == "__main__":
    run_training()