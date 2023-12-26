import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import einops

from utils.arrays import batch_to_device, to_np, to_torch, to_device, apply_dict
from utils.helpers import EMA, soft_copy_nn_module, copy_nn_module, minuscosine
from utils.timer import Timer
from .temporal import TemporalUnet4DD
from .model import MLP
from .diffusion import GaussianDiffusion4DD
from .qnet import CQLCritic, Critic, HindsightCritic

def cycle(dl):
    while True:
        for data in dl:
            yield data
    
    
class DecisionDiffuser(object):
    def __init__(self, 
                 dataset,
                 renderer,
                 goal_dim,
                 device,
                 ## model ##
                 conditional,
                 condition_dropout,
                 calc_energy,
                 ## diffuser ##
                 dim_mults,
                 n_timesteps,
                 clip_denoised,
                 condition_guidance_w,
                 beta_schedule,
                 ## training ##
                #  warmup_steps,
                 maxq=False,
                 alpha=1.0,
                 step_start_ema=1000,
                 ema_decay=0.995,
                 update_ema_every=10,
                 train_batch_size=32,
                 gradient_accumulate_every=5,
                 lr=3e-4,
                 logdir='./logs',
                 diffusion_loadpath='./logs',
                 log_freq=1000,
                 save_freq=10000,
                 sample_freq=1000,
                 label_freq=100000,
                 wandb=False,
                 ):
        state_dim = dataset.observation_dim
        action_dim = dataset.action_dim
        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        self.goal_dim = goal_dim
        self.horizon = dataset.horizon
        
        # self.model = MLP(state_dim, action_dim, goal_dim, dataset.horizon, conditional=conditional, \
        #                 condition_dropout=condition_dropout, calc_energy=calc_energy).to(device)
        self.model = TemporalUnet4DD(self.horizon, self.obsact_dim, goal_dim, conditional=conditional, \
                            dim_mults=dim_mults, condition_dropout=condition_dropout, calc_energy=calc_energy).to(device)
        self.diffuser = GaussianDiffusion4DD(self.model, state_dim, action_dim, goal_dim, self.horizon,\
                                        n_timesteps=n_timesteps, clip_denoised=clip_denoised, \
                                        conditional=conditional, condition_guidance_w=condition_guidance_w, \
                                        beta_schedule=beta_schedule, device=device).to(device)
        self.diffuser_optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=lr)
        self.step_start_ema=step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.diffuser)
        self.update_ema_every = update_ema_every
        
        # self.critic = CQLCritic(state_dim, action_dim, goal_dim, dataset.normalizer).to(device)
        # self.critic = Critic(state_dim, action_dim, goal_dim, dataset.normalizer).to(device)
        self.critic = HindsightCritic(state_dim, action_dim, goal_dim, dataset).to(device)
        self.critic_best = copy.deepcopy(self.critic)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer1 = torch.optim.Adam(self.critic.qf1.parameters(), lr=lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic.qf2.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.critic.actor.parameters(), lr=lr)
        
        self.dataset = dataset
        self.env_name = dataset.env.name
        datalen = len(dataset)
        trainlen = round(datalen*0.8)
        vallen = round(datalen*0.2)
        train, val = torch.utils.data.random_split(dataset, [trainlen, datalen-trainlen], \
                generator=torch.Generator().manual_seed(dataset.seed))

        # self.dataloader = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=False
        # ))
        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            train, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=False, drop_last=True
        ))
        self.dataloader_val = cycle(torch.utils.data.DataLoader(
            val, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=False, drop_last=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False, drop_last=True
        ))
        
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        # self.warmup_steps = warmup_steps
        self.maxq = maxq
        self.alpha = alpha

        self.renderer = renderer
        self.logdir = logdir
        self.diffusion_loadpath = diffusion_loadpath
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.wandb = wandb
        self.step = 0
    
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.diffuser)
    
    def train(self, n_train_steps):
        best_loss_q = 1e0
        for step in range(int(n_train_steps)):
            self.model.train()
            timer = Timer()
            ## Critic
            batch = next(self.dataloader_train)
            batch = batch_to_device(batch)
            if 'Fetch' in self.env_name or 'maze' in self.env_name:
                # any states drawn from D
                goal_rand = batch.trajectories[:, 0, :self.goal_dim].clone()
            else:
                goal_rand = batch.rtgs[:, 0].clone()
            batch = next(self.dataloader_train)
            batch = batch_to_device(batch)
            loss_q1, loss_q2, q, qloss1, qloss2 = self.critic.loss(batch, goal_rand, self.ema_model)
            
            self.critic_optimizer1.zero_grad()
            loss_q1.backward()
            self.critic_optimizer1.step()
            
            self.critic_optimizer2.zero_grad()
            loss_q2.backward()
            self.critic_optimizer2.step()
            
            self.actor_optimizer.zero_grad()
            loss_actor = -q
            loss_actor.backward()
            self.actor_optimizer.step()
            
            loss_q = torch.min(loss_q1, loss_q2)
            
            ## Diffuser
            self.diffuser_optimizer.zero_grad()
            for i in range(self.gradient_accumulate_every):
                # batch = next(self.dataloader_train)
                # batch = batch_to_device(batch)
                # rand_goal = batch.trajectories[:, np.random.randint(self.horizon, size=1), :self.goal_dim].reshape(self.batch_size, -1).clone()
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                observation = batch.trajectories[:, :, :self.observation_dim]
                action = batch.trajectories[:, :, self.observation_dim:]                
                
                # hindsight experience goal/reward
                trajectories = batch.trajectories
                if 'Fetch' in self.env_name or 'maze' in self.env_name:
                    # goal = torch.cat([trajectories[:self.batch_size, -1, :self.goal_dim], rand_goal], 0)
                    goal = trajectories[:, -1, :self.goal_dim]
                    goal_rpt = einops.repeat(goal, 'b d -> b r d', r=self.horizon)
                    values = self.critic.q_min(self.critic.unnorm(observation, 'observations'), 
                                            self.critic.unnorm(action, 'actions'), 
                                            self.critic.unnorm(goal_rpt, 'goals'))
                else:
                    goal = batch.rtgs[:, -1].clone()
                    goal_rpt = einops.repeat(goal, 'b d -> b r d', r=self.horizon)
                    values = self.critic.q_min(self.critic.unnorm(observation, 'observations'), 
                                            self.critic.unnorm(action, 'actions'), 
                                            goal_rpt)
                
                returns = torch.cat([goal, values.mean(1).detach()], dim=-1)
                loss_d = self.diffuser.loss(trajectories, None, returns)
                loss_d.backward()
            self.diffuser_optimizer.step()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
                
            # Step target network
            if self.step % self.update_ema_every == 0:
                self.step_ema()          
                self.critic.target_update()
            
            # Validation
            self.critic.eval()
            batch_val = next(self.dataloader_val)
            batch_val = batch_to_device(batch)
            if 'Fetch' in self.env_name or 'maze' in self.env_name:
                # any states drawn from D
                goal_rand_val = batch_val.trajectories[:, 0, :self.goal_dim].clone()
            else:
                goal_rand_val = batch_val.rtgs[:, 0].clone()
            batch_val = next(self.dataloader_val)
            batch_val = batch_to_device(batch_val)
            with torch.no_grad():
                loss_q1_val, loss_q2_val, _, _, _ = self.critic.loss(batch_val, goal_rand_val, self.ema_model)
            loss_q_val = torch.min(loss_q1_val, loss_q2_val)
            if loss_q_val < best_loss_q:
                print(f'** min val_loss for critic! ')
                best_loss_q = loss_q_val
                copy_nn_module(self.critic, self.critic_best)  

            # save
            if (self.step+1) % self.save_freq == 0:
                label = self.step
                self.save(label)
                
            if self.step % self.log_freq == 0:
                print(f'{self.step}: loss_d: {loss_d:8.4f} | loss_q:{loss_q:8.4f} | q:{q.mean():8.4f} | time:{timer()}', flush=True)

                    
            self.step += 1
    
    def save(self, epoch):
        data = {
            'step': self.step,
            'ema': self.ema_model.state_dict(),
            'critic': self.critic_best.state_dict(),
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        
    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.diffusion_loadpath, f'state_{epoch}.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.ema_model.load_state_dict(data['ema'])
        self.critic.load_state_dict(data['critic'])


    def render_samples(self, rollout, trial):
        '''
            renders samples from (ema) diffusion model
        '''
        # rollout = self.dataset.normalizer.unnormalize(rollout, 'observations')

        savepath = os.path.join(self.logdir, f'sample-{self.step}-{trial}.png')
        self.renderer.composite(savepath, rollout)