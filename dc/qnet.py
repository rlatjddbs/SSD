import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.distributions import Exponential

from utils.helpers import soft_copy_nn_module, minuscosine
from utils.arrays import to_np, to_torch
    
class HindsightCritic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, dataset, hidden_dim=256, gamma=0.99, 
                 min_q_weight=1.0, temp=1.0, n_random=50, max_q_backup=False):
        super(HindsightCritic, self).__init__()
        self.qf1 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1),
                                      )

        self.qf2 = nn.Sequential(nn.Linear(state_dim + action_dim + goal_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1),
                                      )
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)
        self.actor = nn.Sequential(nn.Linear(state_dim+goal_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.Mish(),
                                   nn.Linear(hidden_dim, action_dim),
                                   )

        self.gamma = gamma
        self.n_random = n_random
        self.min_q_weight = min_q_weight
        self.temp = temp
        self.max_q_backup = max_q_backup

        self.observation_dim = state_dim
        self.action_dim = action_dim
        self.obsact_dim = state_dim + action_dim
        self.goal_dim = goal_dim
        
        try:
            self.has_object = dataset.env.has_object
        except:
            self.has_object = False
            
        self.env_name = dataset.env.name
        self.normalizer = dataset.normalizer
        if 'goals' in self.normalizer.normalizers:
            self.goal_key = 'goals'
        elif 'rtgs' in self.normalizer.normalizers:
            self.goal_key = 'rtgs'
            
    def forward(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x), self.qf2(x)
    
    def forward_target(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1_target(x), self.qf2_target(x)
    
    def forward_actor(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return torch.clamp(self.actor(x), -1, 1)
    
    def q1(self, state, action, goal):
        x = torch.cat([state, action, goal], dim=-1)
        return self.qf1(x)

    def q_min(self, state, action, goal):
        q1, q2 = self.forward(state, action, goal)
        return torch.min(q1, q2)
    
    def target_update(self):
        soft_copy_nn_module(self.qf1, self.qf1_target)
        soft_copy_nn_module(self.qf2, self.qf2_target)
    
    def loss(self, batch, goal_rand, ema_model):
        trajectories = batch.trajectories.clone()
        batch_size, horizon, _ = trajectories.shape
        
        ## Hindsight goals and values
        if self.has_object:
            '''
                Concatenate hindsight-relabeled value and unrelabeled value.
                All the (values_cat) are of batch size 2b, where first b is 
                for relabeled value and the other b is unrelabeled value.
                
                future_t: k.
                values: Hindsight relabeled value. gamma ** (k-1)
                next_q: Q(s',a')
                td_target: cat([values, gamma * next_q], 0)
            '''
            
            if 'kitchen' in self.env_name:
                ag = self.unnorm(trajectories[..., :self.observation_dim], 'achieved_goals')
            elif 'Fetch' in self.env_name:
                ag = self.unnorm(trajectories[..., self.goal_dim:2*self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation = self.unnorm(trajectories[:, 0, :self.observation_dim], 'observations')
            action = self.unnorm(trajectories[:, 0, self.observation_dim:], 'actions')
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            future_t = np.random.randint(1, horizon, size=(batch_size,))
            hindsight_goals = ag[np.arange(batch_size), future_t]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            discount = self.gamma ** (future_t - 1)
            values = to_torch((discount * 1).reshape(batch_size, 1))
            next_observation = self.unnorm(trajectories[:, 1, :self.observation_dim], 'observations')
            next_action = self.unnorm(trajectories[:, 1, self.observation_dim:], 'actions')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, self.unnorm(goal_rand, 'goals'))
            
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)
            
            # observation, action, next_observation, next_action, nextnext_observation = self.unnorm_transition(trajectories, self.has_object)
            # nextnext_action = ema_model(nextnext_observation, 
            #                         torch.ones_like(batch.rtgs), 
            #                         batch.goals[:, -1], self.has_object)[:, 0, self.observation_dim:]
            # observation_cat = next_observation.repeat(2,1)
            # action_cat = next_action.repeat(2,1)
            # goals = torch.cat([batch.goals[:, -1], goal_rand], 0)
            # goals = self.unnorm(goals, 'achieved_goals')
            
            # values = torch.ones((batch_size, 1)).to('cuda')
            # next_q1, next_q2 = self.forward_target(nextnext_observation, nextnext_action, goal_rand)
        else:
            '''
                Concatenate hindsight-relabeled value and unrelabeled value.
                All the (values_cat) are of batch size 2b, where first b is 
                for relabeled value and the other b is unrelabeled value.
                
                future_t: k=1.
                values: Hindsight relabeled value. 1.
                next_q: Q(s',a')
                td_target: cat([values, gamma * next_q], 0)
            '''
            ag = self.unnorm(trajectories[:, :, :self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation, action, next_observation, next_action, _ = self.unnorm_transition(trajectories, self.has_object)
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            hindsight_goals = ag[np.arange(batch_size), 1]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            values = torch.ones((batch_size, 1)).to('cuda')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, self.unnorm(goal_rand, 'goals'))
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)

        # calaulate q value
        pred_q1, pred_q2 = self.forward(observation_cat, action_cat, goals_cat)
        
        # For negative action reg
        # x = torch.cat([observation_cat, goals], -1)
        targ_q1, targ_q2 = self.forward_target(observation_cat, action_cat, goals_cat)
        targ_q = torch.min(targ_q1, targ_q2)
        # sample negative action
        num_random_actions = 10
        random_actions = torch.FloatTensor(batch_size * num_random_actions, self.action_dim).uniform_(-1, 1).to(action.device)
        obs_rpt = observation.repeat_interleave(num_random_actions, axis=0)
        goals_rrpt = hindsight_goals.repeat_interleave(num_random_actions, axis=0)
        rand_q1, rand_q2 = self.forward(obs_rpt, random_actions, goals_rrpt)        
        rand_q1 = rand_q1.reshape(batch_size, -1)
        rand_q2 = rand_q2.reshape(batch_size, -1)
        i1 = torch.distributions.Categorical(logits=rand_q1.detach()).sample()
        i2 = torch.distributions.Categorical(logits=rand_q2.detach()).sample()
        
        min_q1_loss = rand_q1[torch.arange(batch_size), i1].mean()
        min_q2_loss = rand_q2[torch.arange(batch_size), i2].mean()
        
        # Final loss
        loss1 = F.mse_loss(td_target1.detach(), pred_q1, reduction='mean') + (min_q1_loss**2).mean()
        loss2 = F.mse_loss(td_target2.detach(), pred_q2, reduction='mean') + (min_q2_loss**2).mean()
        
        return loss1, loss2, targ_q.mean(), (min_q1_loss**2).mean(), (min_q2_loss**2).mean()

    def loss_critic(self, batch, goal_rand, ema_model):
        trajectories = batch.trajectories.clone()
        batch_size, horizon, _ = trajectories.shape
        
        # hindsight goals and values
        if self.has_object:
            # Hindsight goals
            ag = self.unnorm(trajectories[:, :, self.goal_dim:2*self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation = self.unnorm(trajectories[:, 0, :self.observation_dim], 'observations')
            action = self.unnorm(trajectories[:, 0, self.observation_dim:], 'actions')
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            future_t = np.random.randint(1, horizon, size=(batch_size,))
            hindsight_goals = ag[np.arange(batch_size), future_t]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            # Hindsight values
            discount = self.gamma ** (future_t - 1)
            values = to_torch((discount * 1).reshape(batch_size, 1))
            next_observation = self.unnorm(trajectories[:, 1, :self.observation_dim], 'observations')
            sample = ema_model(trajectories[:, 1, :self.observation_dim], 
                               torch.ones_like(batch.rtgs),
                               goal_rand,
                               self.has_object)
            next_action = self.unnorm(sample[:, 0, self.observation_dim:], 'actions')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, dg)
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)
            
            # observation, action, next_observation, next_action, nextnext_observation = self.unnorm_transition(trajectories, self.has_object)
            # nextnext_action = ema_model(nextnext_observation, 
            #                         torch.ones_like(batch.rtgs), 
            #                         batch.goals[:, -1], self.has_object)[:, 0, self.observation_dim:]
            # observation_cat = next_observation.repeat(2,1)
            # action_cat = next_action.repeat(2,1)
            # goals = torch.cat([batch.goals[:, -1], goal_rand], 0)
            # goals = self.unnorm(goals, 'achieved_goals')
            
            # values = torch.ones((batch_size, 1)).to('cuda')
            # next_q1, next_q2 = self.forward_target(nextnext_observation, nextnext_action, goal_rand)
        else:
            # Hindsight goals
            ag = self.unnorm(trajectories[:, :, :self.goal_dim], 'achieved_goals')
            dg = self.unnorm(goal_rand, 'goals')
            observation, action, next_observation, next_action, _ = self.unnorm_transition(trajectories, self.has_object)
            observation_cat = observation.repeat(2,1)
            action_cat = action.repeat(2,1)
            hindsight_goals = ag[np.arange(batch_size), 1]
            goals_cat = torch.cat([hindsight_goals, dg], 0)
            
            # Hindsight values
            values = torch.ones((batch_size, 1)).to('cuda')
            next_q1, next_q2 = self.forward_target(next_observation, next_action, self.unnorm(goal_rand, 'goals'))
            td_target1 = torch.cat([values, self.gamma * next_q1], 0)
            td_target2 = torch.cat([values, self.gamma * next_q2], 0)

        # calaulate q value
        pred_q1, pred_q2 = self.forward(observation_cat, action_cat, goals_cat)
        
        # For negative action reg
        # x = torch.cat([observation_cat, goals], -1)
        targ_q1, targ_q2 = self.forward_target(observation_cat, action_cat, goals_cat)
        targ_q = torch.min(targ_q1, targ_q2)
        # negative_action = self.actor(x)
        # min_q1_loss, min_q2_loss = self.forward(observation_cat, negative_action.detach(), goals)
        # sample negative action
        num_random_actions = 10
        random_actions = torch.FloatTensor(batch_size * num_random_actions, action.shape[-1]).uniform_(-1, 1).to(action.device)
        obs_rpt = observation.repeat_interleave(num_random_actions, axis=0)
        goals_rrpt = hindsight_goals.repeat_interleave(num_random_actions, axis=0)
        rand_q1, rand_q2 = self.forward(obs_rpt, random_actions, goals_rrpt)        
        rand_q1 = rand_q1.reshape(batch_size, -1)
        rand_q2 = rand_q2.reshape(batch_size, -1)
        i1 = torch.distributions.Categorical(logits=rand_q1.detach()).sample()
        i2 = torch.distributions.Categorical(logits=rand_q2.detach()).sample()
        # x = torch.cat([observation, goals_rpt], -1)
        # negative_action = self.actor(x)
        
        # min_q1_loss, min_q2_loss = self.forward(observation, negative_action.detach(), goals_rpt)
        min_q1_loss = rand_q1[torch.arange(batch_size), i1].mean()
        min_q2_loss = rand_q2[torch.arange(batch_size), i2].mean()
        
        # Final loss
        loss1 = F.mse_loss(td_target1.detach(), pred_q1, reduction='mean') + (min_q1_loss**2).mean()
        loss2 = F.mse_loss(td_target2.detach(), pred_q2, reduction='mean') + (min_q2_loss**2).mean()
        
        return loss1, loss2, targ_q.mean(), (min_q1_loss**2).mean(), (min_q2_loss**2).mean()
    
    def unnorm(self, x, key):
        return to_torch(self.normalizer.unnormalize(to_np(x), key))

    def norm(self, x, key):
        return to_torch(self.normalizer(to_np(x), key))
    
    def unnorm_transition(self, trajectories, has_object):
        observation = self.unnorm(trajectories[:, 0, :self.observation_dim], 'observations')
        action = self.unnorm(trajectories[:, 0, self.observation_dim:], 'actions')
        next_observation = self.unnorm(trajectories[:, 1, :self.observation_dim], 'observations')
        next_action = self.unnorm(trajectories[:, 1, self.observation_dim:], 'actions')
        if has_object:
            nextnext_observation = self.last_obs(next_observation)
        else:
            nextnext_observation = next_observation[:, -1]

        return observation, action, next_observation, next_action, nextnext_observation

    def make_indices(self, batch_size, horizon, relabel_percent=0.5):
        her_indexes = np.where(np.random.uniform(size=batch_size) < relabel_percent)
        t_indexes = (np.random.randint(horizon, size=batch_size))[her_indexes]
        return her_indexes, t_indexes
    
    def last_obs(self, obs):
        last_obs = obs.clone()
        last_obs[:, :self.goal_dim] = obs[:, self.goal_dim:2*self.goal_dim]
        last_obs[:, 2*self.goal_dim:3*self.goal_dim] = torch.zeros_like(last_obs[:, 2*self.goal_dim:3*self.goal_dim])
        last_obs[:, 3*self.goal_dim:] = 0
        return last_obs
