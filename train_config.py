import torch
from math import sqrt

class Config:
    seed = 54
    img_size = 512
    classes = 11014
    scale = 30
    margin = 0.5
    fc_dim = 512
    epochs = 18
    batch_size = 8
    num_workers = 8
    model_name = 'tf_efficientnet_b3'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    scheduler_params = {
        "lr_start": 1e-5 * sqrt(2),
        "lr_max": 1e-5 * 32,     # 1e-5 * 32 (if batch_size(=32) is different then)
        "lr_min": 1e-6 * sqrt(2),
        "lr_ramp_ep": 5,
        "lr_sus_ep": 0,
        "lr_decay": 0.8,
    }