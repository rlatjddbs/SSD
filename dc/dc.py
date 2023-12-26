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
from utils.eval_module import main, main_maze, main_kitchen
from dc.policy import *
from .temporal import TemporalUnetTransformer
from .model import MLP
from .diffusion import GaussianDiffusion
from .qnet import HindsightCritic

def cycle(dl):
    while True:
        for data in dl:
            yield data
    

def pad_horizon(x, horizon):
    '''
        x : shape of (b horizon d)
        out: shape of (b 2*horizon d)
    '''
    x = einops.rearrange(x, 'b h d -> b d h')
    x = F.pad(x, (0, horizon), mode='replicate')
    return einops.rearrange(x, 'b d h -> b h d')
    

class DiffuserCritic(object):
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
                 action_weight,
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
        # self.model = TemporalUnetConditional(self.horizon, self.obsact_dim, goal_dim, conditional=conditional, \
        #                     dim_mults=dim_mults, condition_dropout=condition_dropout, calc_energy=calc_energy).to(device)
        self.model = TemporalUnetTransformer(self.horizon, self.obsact_dim, goal_dim, conditional=conditional, \
                            dim_mults=dim_mults, condition_dropout=condition_dropout, calc_energy=calc_energy).to(device)
        self.diffuser = GaussianDiffusion(self.model, state_dim, action_dim, goal_dim, self.horizon,\
                                        n_timesteps=n_timesteps, clip_denoised=clip_denoised, action_weight=action_weight,\
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
        try:
            self.has_object = dataset.env.has_object
        except:
            self.has_object = False
            
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
        env = self.dataset.env
        best_loss_q = 1e0
        for step in range(int(n_train_steps)):
            self.model.train()
            timer = Timer()
            """
                Critic
            """
            ## Sample random goal which will be used to relabel.
            batch = next(self.dataloader_train)
            batch = batch_to_device(batch)
            # if 'Fetch' in self.env_name or 'maze' in self.env_name:
            goal_rand = batch.goals[:, 0].clone()
            # else:
            #     goal_rand = batch.rtgs[:, 0].clone()
                
            ## Calculate critic loss and step.
            batch = next(self.dataloader_train)
            batch = batch_to_device(batch)
            loss_q1, loss_q2, q, qloss1, qloss2 = self.critic.loss(batch, goal_rand, self.ema_model)
            
            self.critic_optimizer1.zero_grad()
            loss_q1.backward()
            self.critic_optimizer1.step()
            
            self.critic_optimizer2.zero_grad()
            loss_q2.backward()
            self.critic_optimizer2.step()
            
            loss_q = torch.min(loss_q1, loss_q2)
            
            
            """
                Diffuser
            """
            
            self.diffuser_optimizer.zero_grad()
            for i in range(self.gradient_accumulate_every):
                '''
                    Sample trajectories of observation and action.
                '''
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                observation = batch.trajectories[:, :, :self.observation_dim]
                action = batch.trajectories[:, :, self.observation_dim:]                
                
                '''
                    Hindsight experience goals/values. 
                    Use only successful trajectories to calculate Q values.
                    
                    ag:                     Achieved goal. (b, d)
                    ag_rpt:                 (b, h, d)
                    dg_rpt:                 Desired goal. (b, h, d)
                    current_t:              Randomly selected starting point t.
                    indexes:                From (current_t) to (current_t + horizon).
                    trajectories_padded:    Pad with achieved goal, with shape (b 2h d).
                '''
                if 'kitchen' in self.env_name:
                    ag = batch.trajectories[:, -1, :self.observation_dim] 
                else:
                    ag = batch.trajectories[:, -1, self.goal_dim:2*self.goal_dim] if self.has_object else batch.trajectories[:, -1, :self.goal_dim]
                dg_rpt = batch.goals
                ag_rpt = einops.repeat(ag, 'b d -> b h d', h=self.horizon)
                goals = torch.cat([ag, dg_rpt[:, 0]], 0)
                
                current_t = np.random.randint(0, self.horizon-1, size=(self.batch_size,)) 
                indexes = current_t.reshape((self.batch_size, 1)) \
                        + einops.repeat(np.arange(self.horizon), 'h -> b h', b=self.batch_size)  
                        
                values_ag = self.critic.q_min(self.critic.unnorm(observation, 'observations'), 
                                        self.critic.unnorm(action, 'actions'), 
                                        self.critic.unnorm(ag_rpt, 'achieved_goals'))[np.arange(self.batch_size), current_t]
                values_dg = self.critic.q_min(self.critic.unnorm(observation, 'observations'), 
                                        self.critic.unnorm(action, 'actions'), 
                                        self.critic.unnorm(dg_rpt, 'goals'))[np.arange(self.batch_size), current_t]
                # values_ag_padded = pad_horizon(values_ag, self.horizon)
                # values_dg_padded = pad_horizon(values_dg, self.horizon)
                # values = torch.cat([values_ag_padded[np.arange(self.batch_size).reshape(self.batch_size, 1), indexes], \
                #                     values_dg_padded[np.arange(self.batch_size).reshape(self.batch_size, 1), indexes]], 0)
                values = torch.cat([values_ag, values_dg], 0)
                
                trajectories_padded = pad_horizon(batch.trajectories, self.horizon)
                trajectories_padded = trajectories_padded[np.arange(self.batch_size).reshape(self.batch_size, 1), indexes]
                
                # agdg
                trajectories_padded = trajectories_padded.repeat((2, 1, 1))
                
                # ag
                # loss_d = self.diffuser.loss(trajectories_padded, values_ag.detach(), ag, has_object=self.has_object)
                # agdg
                loss_d = self.diffuser.loss(trajectories_padded, values.detach(), goals, has_object=self.has_object)
                loss_d = loss_d / self.gradient_accumulate_every
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
                goal_rand_val = batch_val.goals[:, 0].clone()
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
                output = {
                        "loss_d": loss_d,
                        "loss_q": loss_q,
                        "loss_q_val": loss_q_val,
                        "Q": q.mean(),
                        "qloss1": qloss1,
                        "qloss2": qloss2, }
                if 'Fetch' in self.env_name:
                    policy = FetchControl(self.ema_model, self.dataset.normalizer, self.observation_dim, self.goal_dim, self.has_object)
                    succ_rates, undisc_returns, disc_returns, distances = main(env, 10, policy, 1.)
                    output["success_rate"] = np.array(succ_rates).mean()
                    output["returns"] = np.array(undisc_returns).mean()
                    output["discounted_returns"] = np.array(disc_returns).mean()
                    output["distance"] = np.array(distances).mean()
                # elif 'maze2d' in self.env_name:
                #     policy = GoalPositionControl(self.ema_model, self.dataset.normalizer, self.observation_dim, self.goal_dim, self.has_object)
                #     succ_rates, undisc_returns, scores, distances = main_maze(env, 10, policy, 1.)
                #     output["success_rate"] = np.array(succ_rates).mean()
                #     output["returns"] = np.array(undisc_returns).mean()
                #     output["scores"] = np.array(scores).mean()
                #     output["distance"] = np.array(distances).mean()
                
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