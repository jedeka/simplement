"""Deepseek R1 Recreation
https://github.com/huggingface/open-r1/tree/main
"""

import numpy as np 
import torch 
from torch import nn 
import torch.nn.functional as F

# ***** utils ***** 
# NOTE: no need sys.path hacks if running from main dir
# import sys; sys.path.append('..') 
from simplement.models.base import BaseModel # just absolute path

