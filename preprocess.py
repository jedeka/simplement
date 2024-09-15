"""
Transformer torch reimplementation
Refs:
- https://github.com/jadore801120/attention-is-all-you-need-pytorch  
"""
# import torch.multiprocessing as mp
# if mp.get_start_method(allow_none=True) is None:
#     mp.set_start_method('spawn', force=True)  # or 'forkserver'

import argparse, os, time, random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange 

import torch
from torchvision.transforms import transforms

import arguments

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    arguments.add_common_args(parser)
    arguments.add_model_args(parser)
    

    

class Transformer:
    pass 