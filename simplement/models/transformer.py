"""Transformer
TODO:
- All components and visualization 
"""

import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MHA(nn.Module):
    """Multi-Head Attention"""
    def __init__():
        super().__init__()

    @staticmethod
    def sdpa(self, q, k ,v):
        """Scaled Dot-Product Attention"""

    def forward(self, x):
        pass 

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 

class CausalTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass 
