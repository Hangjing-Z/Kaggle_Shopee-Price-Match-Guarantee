import os
import cv2
import math
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

def read_test_dataset():
    df = pd.read_csv('./input/shopee-folds/folds-random-10.csv')
    df = df[df['fold']==0].reset_index(drop=True) # test
    
    df_cu = pd.DataFrame(df)
    image_paths = './input/shopee-product-matching/train_images/' + df['image']

    return df, df_cu, image_paths

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def scores(y_true, y_pred):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=10
    '''
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    precision = intersection / len_y_pred 
    recall =  intersection / len_y_true
    return f1 , precision , recall

def train_fn(model, data_loader, optimizer, scheduler, epoch, device):
    '''
    Reference: https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images?scriptVersionId=58269290&cellId=24
    '''
    model.train()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Training epoch: " + str(epoch+1))

    for t,data in enumerate(tk):
        optimizer.zero_grad()
        for k,v in data.items():
            data[k] = v.to(device)

        _, loss = model(**data)
        loss.backward()
        optimizer.step() 
        fin_loss += loss.item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})

    scheduler.step()
    return fin_loss / len(data_loader)


def eval_fn(model, data_loader, epoch, device):
    '''
    Reference: https://www.kaggle.com/tanulsingh077/pytorch-metric-learning-pipeline-only-images?scriptVersionId=58269290&cellId=26
    '''
    model.eval()
    fin_loss = 0.0
    tk = tqdm(data_loader, desc = "Validation epoch: " + str(epoch+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(device)

            _, loss = model(**data)
            fin_loss += loss.item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
    return fin_loss / len(data_loader)
