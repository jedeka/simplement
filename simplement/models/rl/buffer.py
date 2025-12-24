import numpy as np

import torch
from torch import nn 
import torch.nn.functional as F


class RLBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = cfg 

        
