import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class MLP(nn.Module):
    
    def __init__(
        self, 
        state_dim,
        action_dim, 
        goal_dim,
        horizon,
        time_dim=16, 
        conditional=False, 
        condition_dropout=0.1, 
        calc_energy=False
        ):
        super(MLP, self).__init__()
        
        transition_dim = state_dim*2 + action_dim + 2
        
        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()
            
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim*2),
            act_fn,
            nn.Linear(time_dim*2, time_dim)
        )
        
        self.conditional = conditional
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        
        if self.conditional:
            self.cond_layer = nn.Sequential(
                nn.Linear(goal_dim+1, 256),
                act_fn,
                nn.Linear(256, 256),
                act_fn,
                nn.Linear(256, transition_dim)
            )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            input_dim = transition_dim * horizon + time_dim + transition_dim
        else:
            input_dim = transition_dim * horizon + time_dim 
            
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            act_fn, 
            nn.Linear(256, 256),
            act_fn,
            nn.Linear(256, 256),
            act_fn
        )
        output_dim = transition_dim * horizon - state_dim
        self.final_layer = nn.Linear(256, output_dim)
        
    def forward(self, x, time, state, goal, use_dropout=True, force_dropout=False):
        if self.conditional:
            assert goal is not None
            cond_embed = self.cond_layer(goal)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=tuple(cond_embed.shape)).to(cond_embed.device)
                cond_embed = mask*cond_embed
            if force_dropout:
                cond_embed = 0*cond_embed
            
            t = self.time_mlp(time)
            x = torch.cat([x, t, state, cond_embed], dim=1)
        else:
            t = self.time_mlp(time)
            x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        
        return self.final_layer(x)