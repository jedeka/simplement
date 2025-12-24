import numpy as np
from collections import deque
from copy import deepcopy
import random, math 
from tqdm import trange

import torch
from torch import nn 
import torch.nn.functional as F
from torch import optim

from ..base import RLBaseModel, RLBaseAgent 


def preprocess_obs(obs, device):
    return torch.tensor(obs, dtype=torch.float).unsqueeze(0).to(device)

class ReplayBuffer:
    """Buffer for experience replay"""
    def __init__(self, max_size=50000, batch_size=64):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)
    
    def check(self):
        print(self.buffer)

    def push(self, experience: list):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

class DQN(RLBaseModel):
    def __init__(self, cfg, is_image=True):
        super().__init__(cfg)

        if is_image:
            self.feature_net = nn.Sequential(
                # nn.Linear(self.state_dim, self.n_embd),
                nn.Conv2d(self.state_dim[-1], 16, kernel_size=2),
                # nn.MaxPool2d((2,2)),
                nn.ReLU(),  
                nn.Conv2d(16, 32, kernel_size=2),
                nn.ReLU(),  
                # nn.Flatten()
            )
        else:
            self.feature_net = nn.Sequential(
                nn.Linear(self.state_dim[-1], 32),
                nn.ReLU(), 
                nn.Linear(32, 128),
                nn.ReLU(), 
                nn.Linear(128, 128),
                nn.ReLU(), 
                nn.Linear(128, 32),
            )
        
        # auto assign size for flatten output 
        B, C, H, W = self.feature_net(torch.zeros(*self.state_dim).unsqueeze(0).permute(0, -1, -3, -2)).shape
        flatten_size = C * H * W
        self.fc = nn.Sequential(
            nn.Linear(flatten_size, 256), 
            nn.ReLU(inplace=True), 
            nn.Linear(256, self.action_dim),
        )

    def forward(self, obs):
        x = obs.permute(0, -1, -3, -2) # if len(obs.shape) == 4 else obs.permute(-1, -3, -2)
        x = self.feature_net(x).flatten(1, -1) # flatten every dim except batch
        x = self.fc(x)
        return x


class DQNAgent(RLBaseAgent):
    """Deep Q Networks. Limited to one env each init 
    Args:
    - cfg: config
    """
    def __init__(self, cfg, env, is_image=True):
        super().__init__(cfg)
        self.env = env
        self.policy_net = DQN(cfg, is_image=is_image)
        self.target_net = deepcopy(self.policy_net)
        self.buffer = ReplayBuffer(cfg['max_buffer_len'])
        self.optim = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.steps = 0
        self.reset()

    def reset(self):
        obs, info = self.env.reset()
        # TODO: hypothesis: can't be reset, 
        ## else we will stuck too long in the same epsilon rate
        if self.reset_timesteps:
            self.steps = 0
        return obs, info 
    
    @torch.no_grad()
    def act(self, state): 
        eps_start, eps_end = self.cfg['eps_start'], self.cfg['eps_end']
        eps_decay = self.cfg['eps_decay']
        eps_threshold = eps_end + (eps_start - eps_end) *  \
            math.exp(-1 *  self.steps / eps_decay)
        # eps_delta = (eps_start - eps_end) / (self.steps + 1e-6)
        
        self.steps += 1 
        # eps_threshold = 0.2
        if np.random.rand() < eps_threshold:
            action = np.random.choice(self.cfg['action_dim'])
        else:
            # print(state.shape, 'in act')
            q_value = self.policy_net(state)
            action = q_value.argmax(-1).item()
    
        return action, eps_threshold
    
    def optimize(self, batch):
        """Optimize and compute DQN loss for a batch sampled from the buffer
        NOTE:
        - seems not working without final state mask. Q: How?
        """
        if len(batch) < self.batch_size:
            return None 

        states, actions, rewards, next_states, dones = [], [], [], [], []
        # prep work to parallelize the batch  
        for b in batch:
            state, action, reward, next_state, done = b
            states.append(state); next_states.append(next_state)
            actions.append(action); rewards.append(reward), dones.append(done)
   
        states = torch.cat(states)
        next_states = torch.cat(next_states)
        actions = torch.tensor(actions).to(self.device).unsqueeze(-1)    
        rewards = torch.tensor(rewards).to(self.device)#.unsqueeze(-1)
        dones = torch.FloatTensor(dones).to(self.device)#.unsqueeze(-1)

        # gather q_values for actions 
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1)[0]
        td_target = rewards + self.discount_factor * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, td_target.unsqueeze(-1))
        

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return loss.item() 
        
    def learn(self, timesteps=None):
        """Training loop for 1 episode"""
        losses, eps, rewards = [], [], []
    
        timesteps = self.max_timesteps if timesteps is None else timesteps

        obs, _ = self.reset() 
        done = False 
    
        # for t in trange(self.max_timesteps, leave=False):
        while not done:
            state = preprocess_obs(obs, self.device)
            action, curr_eps = self.act(state)

            ret = self.env.step(action)
            if len(ret) > 4: # a lazy compatibility workaround  
                next_obs, reward, terminated, truncated, info = ret 
            else:
                next_obs, reward, terminated, info = ret 
            done = terminated or truncated

            next_state = preprocess_obs(next_obs, self.device)
            self.buffer.push([state, action, reward, next_state, done]) # push a batch 
            
            batch = []
            if len(self.buffer) % self.sample_freq == 0:
                batch = self.buffer.sample(batch_size=self.batch_size)
            
            loss = self.optimize(batch)

            losses.append(0.0 if loss is None else loss)
            eps.append(curr_eps)
            rewards.append(reward)
            
            if done:
                break 
            obs = next_obs
        
        # return info
        info = {
            'losses': losses, # :list, 
            'epsilons': eps, # :float, 
            'rewards': rewards, # :list
        }
        return info 
            
    @torch.no_grad()
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # for key in self.policy_net.state_dict():
        #     self.target_net.state_dict[key] = self.policy_net.state_dict[key] * self.tau + \
        #         self.target_net.state_dict[key] * (1-self.tau)

    def save_model(self, fn):
        self.policy_net.save_model(f'policy_{fn}')
        # self.save_model(fn)
    
    def load_model(self, fn):   
        # self.load_model(fn)
        self.policy_net.load_model(fn)
