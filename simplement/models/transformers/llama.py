"""Transformer
TODO:
- All components and visualization 
"""
import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F

# ***** utils ***** 
# NOTE: no need sys.path hacks if running from main dir
# import sys; sys.path.append('..') 
from simplement.models.base import BaseModel # just absolute path

class LlamaConfig:
    pass 

# ***** models *****
class Llama(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

from transformers import LlamaConfig
# if __name__ == '__main__':
    # import sys 
    # sys.path.append('..')

    # from base import BaseModel
    # print(BaseModel) 
