import torch
import numpy as np

class ScheduledOptim:
    """Simple wrapper class for learning rate scheduling""" 
    def __init__(self, optimizer, lr_mul, d_model, warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.warmup_steps = warmup_steps    
        self.n_steps = 0 

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model 
        n_steps, n_warmup_steps = self.n_steps, self.warmup_steps
        
        # TODO: why like this?
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))
    
    def _update_learning_rate(self):
        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            