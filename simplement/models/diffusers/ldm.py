"""Diffusion models
TODO:
- DDPM
- DDIM
- LDM
- VDM 
- DiT 
- Diffusion policy
"""

import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F
import einops


from models.base import Base, BaseModel

# ***** utils ***** 

# ***** models *****
class DDPM(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

class DDIM(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

class LDM(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

class DiT(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

class DiffusionPolicy(BaseModel):
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)

