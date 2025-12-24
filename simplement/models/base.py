import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F


class Base(nn.Module):
    """Abstract Base Class, for config related base init"""
    def __init__(self, cfg, verbose=False):
        super().__init__()
        self.name = self.__class__.__name__
        if verbose: print(f'Using {self.name}')
        
        self.cfg = cfg
        for k, v in self.cfg.items():
            setattr(self, k, v)

class BaseEnv(Base):
    """Abstract Base Class, for envs"""
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)
        if 'dataset' in cfg.keys():
            for k, v in self.cfg['dataset'].items():
                setattr(self, k, v)
                
class BaseTrainer(Base):
    """Abstract Base Class, for envs"""
    def __init__(self, cfg, verbose=False):
        super().__init__(cfg, verbose)
        if 'train' in cfg.keys():
            for k, v in self.cfg['train'].items():
                setattr(self, k, v)
                

class BaseModel(Base):
    """Abstract Base Class, for DL model"""
    def __init__(self, cfg, use_state=False, verbose=False):    
        super().__init__(cfg, verbose)
        if 'model' in cfg.keys():
            for k, v in self.cfg['model'].items():
                setattr(self, k, v)
                
    
    def save_model(self, fn, verbose=False):
        if verbose:
            print(f'[INFO] Saving model checkpoint {fn}')
        torch.save({'state_dict': self.state_dict(), 'cfg': self.cfg}, fn)

    def load_model(self, fn, verbose=False):   
        if verbose:        
            print(f'[INFO] Loading model checkpoint {fn}')
        ckpt = torch.load(fn)
        # last_episode = checkpoint['episode']
        state_dict, cfg = ckpt['state_dict'], ckpt['cfg']
        self.load_state_dict(state_dict)
        return cfg
    