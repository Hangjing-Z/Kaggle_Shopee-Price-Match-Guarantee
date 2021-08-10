import torch
from math import sqrt

class Config:
    seed = 54
    classes = 11014 
    scale = 30 
    margin = 0.5
    model_name =  'tf_efficientnet_b3'
    fc_dim = 512
    img_size = 512
    batch_size = 20
    num_workers = 4
    threshold_image=4.5
    threshold_text=0.75
    device =  'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path = './models/effb3_lr1e5sq2_decay8e-1_warmup5_batch8_epoch18_17.pt'