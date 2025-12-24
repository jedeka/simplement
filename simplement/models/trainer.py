"""Trainer module.
"""
import argparse, os, time, random
# this_filename = os.path.basename(__file__)
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange 
from datetime import timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizer, BertModel

# from dataset import Dataset

import pickle

import numpy as np
import torch

# from utils import convert_to_tensor

# from utils import device 
# from utils import DEVICE
# device = torch.device(f'cuda:{DEVICE}' if torch.cuda.is_available() else 'cpu')


# ***** Helpers *****
from utils import TIMESTAMP as timestamp, DEVICE, create_folder
from utils import set_dataset_filename, set_model_filename
from time import perf_counter as pc
create = False # bool to create folder or not

# TODO: put all helper functions below, such as compare model params, etc. 


def assert_cuda(module):
    for name, param in module.named_parameters():
        assert param.device == device, f"Param [{name}] is in device [{param.device}] not in [{device}]"
    print(f'Asserting done, all layers are set to device: [{device}]')


from models.base import Base
class Trainer(Base):
    """TODO:should it be config free?"""
    def __init__(self, model, train_dl, test_dl, optimizer, criterion, logger=None):
        super().__init__(cfg)
        # cfg_env, cfg_dataset = cfg['env'], cfg['dataset']
        # cfg_train = cfg['train']
        self.model = model 
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger

    def train_one_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dl), desc=f'Epoch {epoch} training batch', colour='green', total=len(train_dl)):
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch['optimal_actions']
            
            # for timestep embedding . seqlen: horizon + padding
            batch_size = batch['context_states'].shape[0]
            timesteps = torch.arange(horizon+1).unsqueeze(0).expand(batch_size, horizon+1).to(device)
            batch.update({'timesteps': timesteps}) 

            info = model(batch)
            pred_actions = info['pred_actions']
            
            true_actions = true_actions.reshape(-1, action_dim)
            # pred_actions = pred_actions.reshape(-1, action_dim)
            # print(true_actions.shape, pred_actions.shape)
                    
            loss_dict = loss_fn(info, true_actions)

            action_loss = loss_dict['recon'] # real data prediction loss 
            

    def validate_one_epoch(self):
        pass 


class Evaluator(Base):
    pass 

    

        