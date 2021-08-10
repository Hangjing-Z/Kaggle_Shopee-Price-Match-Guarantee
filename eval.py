import sys
sys.path.append('./input/pytorch-image-models-master')
import timm

import os
import cv2
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
import gc
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from test_config import Config
from preprocess import get_train_transforms, get_test_transforms
from utils import read_test_dataset, seed_torch, scores
from effmodel import ArcMarginProduct, ShopeeScheduler, ShopeeModel

MODEL_PATH = './models/'

seed_torch(Config.seed)

def combine_predictions(row):
    x = np.concatenate([row['image_predictions'], row['text_predictions']])
    return ' '.join( np.unique(x) )

class ShopeeDataset(Dataset):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=16
    '''
    def __init__(self, image_paths, transforms=None):
        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']
        
        return image, torch.tensor(1)

def get_image_embeddings(image_paths):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=17
    '''
    model = ShopeeModel(pretrained=False).to(Config.device)
    state = torch.load(Config.model_path, map_location=Config.device)
    model.load_state_dict(state)
    model.eval()

    image_dataset = ShopeeDataset(image_paths=image_paths, transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )

    embeds = []
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.to(Config.device)
            label = label.to(Config.device)
            features = model(img,label)
            image_embeddings = features.detach().cpu().numpy()
            embeds.append(image_embeddings)

    del model
    image_embeddings = np.concatenate(embeds)
#     print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings


def get_image_neighbors(df, threshold, embeddings, KNN=50):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=14
    '''
#     print(type(embeddings))
    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
#     threshold = 4.0
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions


def get_text_predictions(df, threshold, max_features=25_000, KNN=50):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=19
    '''
    model = TfidfVectorizer(stop_words='english',
                            binary=True,
                            max_features=max_features)
    text_embeddings = model.fit_transform(df_cu['title']).toarray() #.get()
    
    CHUNK = 1024 * 4
    CTS = len(df) // CHUNK
    if (len(df)%CHUNK) != 0:
        CTS += 1

    preds = []
    for j in range( CTS ):
        a = j * CHUNK
        b = (j+1) * CHUNK
        b = min(b, len(df))

        cts = np.matmul(text_embeddings, text_embeddings[a:b].T).T
        for k in range(b-a):
            IDX = np.where(cts[k,]>threshold)[0]
            o = df.iloc[IDX].posting_id.values
            preds.append(o)

    del model,text_embeddings
    gc.collect()
    return df, preds

def get_text_neighbors(df, threshold, embeddings, KNN=50):
    '''
    Reference: https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728?scriptVersionId=59449258&cellId=19
    '''
#     print(type(embeddings))
    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
#     threshold = 0.7
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k,idx]
        posting_ids = df['posting_id'].iloc[ids].values
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions


if __name__ == "__main__":
    threshold_image=Config.threshold_image
    threshold_text=Config.threshold_text

    Config.model_path =  sys.argv[1]

    df,df_cu,image_paths = read_test_dataset()

    image_embeddings = get_image_embeddings(image_paths.values)
    df, image_predictions = get_image_neighbors(df, threshold_image, image_embeddings, KNN=50 if len(df)>3 else 3)
    tdf, text_predictions = get_text_predictions(df, threshold_text, max_features=25_000, KNN=50 if len(df)>3 else 3)
    tmp = df.groupby(['label_group'])['posting_id'].unique().to_dict()
    df['matches'] = df['label_group'].map(tmp)
    df['matches'] = df['matches'].apply(lambda x: ' '.join(x))
    df['image_predictions'] = image_predictions
    df['text_predictions'] = text_predictions
    df['pred_matches'] = df.apply(combine_predictions, axis=1)
    df['f1'], df['precision'] , df['recall'] = scores(df['matches'], df['pred_matches'])
    f1, precision, recall= df['f1'].mean(), df['precision'].mean() , df['recall'].mean() 
    
    print("Threshold-image: %4f  Threshold-text: %4f F1: %4f Precision: %4f Recall: %f"%(threshold_image,threshold_text,f1,precision,recall))
