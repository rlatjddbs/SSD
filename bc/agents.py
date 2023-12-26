import torch
import copy
import wandb
import gym
import numpy as np

from datasets.buffer import ReplayBuffer
from datasets.normalization import DatasetNormalizer
from datasets.preprocessing import get_preprocess_fn
from utils.helpers import EMA
from datasets.d4rl import sequence_dataset, load_environment
from dc.qnet import SimpleCritic
from utils.arrays import batch_to_device, to_torch, to_np

# For fetch
def discounted_return(rewards, gamma, reward_offset=True):
    N, T = rewards.shape[0], rewards.shape[1]
    # if type(rewards[0]) == np.ndarray and len(rewards[0]):
    #     rewards = np.array(rewards).T
    # else:
    #     rewards = np.array(rewards).reshape(1, L)

    if reward_offset:
        rewards += 1   # positive offset

    discount_weights = np.power(gamma, np.arange(T)).reshape(1, T)
    dis_return = (rewards * discount_weights).sum(axis=1)
    undis_return = rewards.sum(axis=1)
    return dis_return, undis_return

def cycle(dl):
    while True:
        for data in dl:
            yield data

class BC(object):
    def __init__(self,
                 dataset,
                 critic,
                 renderer,
                 device,
                 # ema
                 step_start_ema=1000,
                 ema_decay=0.995,
                 update_ema_every=10,
                 train_batch_size=32,
                 gradient_accumulate_every=5,
                 lr=3e-4,
                 logfreq=1000,
                 load_path='/ext_hdd/sykim/DC/logs',
                 wandb=False,
                 ):
        self.observation_dim = dataset.observation_dim
        self.action_dim = dataset.action_dim
        self.goal_dim = 2
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dim+self.goal_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.dataset = dataset
        self.horizon = dataset.horizon
        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
        ))

        self.batch_size = train_batch_size
        
        self.gradient_accumulate_every = gradient_accumulate_every
        self.renderer = renderer
        self.logfreq = logfreq
        self.wandb = wandb
        self.step = 0
        
        if device == 'cuda':
            self.actor.cuda()
            self.ema_model.cuda()
    
    def train(self, n_train_steps):
        for step in range(int(n_train_steps)):
            # Actor
            self.actor_optimizer.zero_grad()
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                s = batch.trajectories[:, 0, :self.observation_dim]
                a = batch.trajectories[:, 0, self.observation_dim:]
                g = batch.goals[:,0]
                loss_a = self.criterion(self.actor(torch.cat([s,g], -1)), a)
                loss_a = loss_a / self.gradient_accumulate_every
                loss_a.backward()
            self.actor_optimizer.step()
                
            if self.step % self.logfreq == 0:
                output = {"loss_q": 0,
                          "q_her": 0,
                          "q_ner": 0,
                          "loss_a": loss_a}
            
            self.step += 1
            
            if self.step % 1000 == 0:
                print(f'Evaluation... {step}/{n_train_steps}')
                with torch.no_grad():
                    total_return = 0
                    for epi in range(10):
                        env = self.dataset.env
                        state = env.reset()
                        env.set_target()
                        goal = env.get_target()
                        epi_return = 0
                        at_goal = False
                        for _ in range(env.max_episode_steps):
                            at_goal = state[:self.goal_dim]
                            action = self.actor(torch.cat([to_torch(state), 
                                                           to_torch(goal)], -1))
                            next_state, reward, done, info = env.step(to_np(action))
                            epi_return += reward
                            if done: break
                            state = next_state
                        total_return += epi_return
                        print('epi return: ', epi_return)
           
class BCCritic(object):
    def __init__(self,
                 dataset,
                 critic,
                 renderer,
                 device,
                 # ema
                 step_start_ema=1000,
                 ema_decay=0.995,
                 update_ema_every=10,
                 train_batch_size=32,
                 gradient_accumulate_every=5,
                 lr=3e-4,
                 logfreq=1000,
                 load_path='/ext_hdd/sykim/DC/logs',
                 wandb=False,
                 ):
        self.observation_dim = dataset.observation_dim
        self.action_dim = dataset.action_dim
        self.goal_dim = 2
        
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dim+self.goal_dim+1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
        self.critic = critic
        self.qnet = SimpleCritic(self.observation_dim, self.action_dim, 2, dataset.normalizer)
        self.qnet_optimizer1 = torch.optim.Adam(self.qnet.qf1.parameters(), lr=lr)
        self.qnet_optimizer2 = torch.optim.Adam(self.qnet.qf2.parameters(), lr=lr)
        
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.dataset = dataset
        self.horizon = dataset.horizon
        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
        ))

        self.batch_size = train_batch_size
        
        self.gradient_accumulate_every = gradient_accumulate_every
        self.renderer = renderer
        self.logfreq = logfreq
        self.wandb = wandb
        self.step = 0
        
        if device == 'cuda':
            self.actor.cuda()
            self.ema_model.cuda()
            self.qnet.cuda()
        
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
    
    def train(self, n_train_steps):
        for step in range(int(n_train_steps)):
            self.actor.train()
            
            # Critic
            batch_her = next(self.dataloader_train)
            batch_her = batch_to_device(batch_her)
            batch_ner = next(self.dataloader_train)
            batch_ner = batch_to_device(batch_ner)
            if self.critic == 'HER':
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=batch_her.trajectories, 
                                          goal=batch_her.trajectories[:, 1, :self.goal_dim], 
                                          tdtarget=torch.ones((self.batch_size, 1), device='cuda'), 
                                          actor=self.actor)
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_ner.goals[:, 0],
                                          tdtarget=batch_ner.rewards[:, 0],
                                          actor=self.actor)
            elif self.critic == 'AM':
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=batch_her.trajectories,
                                          goal=batch_her.trajectories[:, 1, :self.goal_dim],
                                          tdtarget=torch.ones((self.batch_size, 1), device='cuda'),
                                          actor=self.actor)
                value = self.qnet.q_min(batch_ner.trajectories[:, 1, :self.observation_dim], 
                                        batch_ner.trajectories[:, 1, self.observation_dim:],
                                        batch_her.goals[:, 0])
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_her.goals[:, 0],
                                          tdtarget=self.qnet.gamma * value,
                                          actor=self.actor)
            elif self.critic == 'SSD':
                k = torch.randint(2, self.horizon+1, (self.batch_size,), device='cuda')
                trajectory_temp = torch.zeros_like(batch_her.trajectories[:,:2])
                for i, j in enumerate(k):
                    trajectory_temp[i] = batch_her.trajectories[i, self.horizon-j:self.horizon-j+2]
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=trajectory_temp,
                                          goal=batch_her.trajectories[:, -1, :self.goal_dim],
                                          tdtarget=self.qnet.gamma**(k-2) * torch.ones((self.batch_size, 1), device='cuda'),
                                          actor=self.actor)
                value = self.qnet.q_min(batch_ner.trajectories[:, 1, :self.observation_dim], 
                                        batch_ner.trajectories[:, 1, self.observation_dim:],
                                        batch_her.goals[:, 0])
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_her.goals[:, 0],
                                          tdtarget=self.qnet.gamma * value,
                                          actor=self.actor)
            loss_q1, loss_q2 = 0.5*(loss1_her+loss1_ner), 0.5*(loss2_her+loss2_ner)
            
            self.qnet_optimizer1.zero_grad()
            loss_q1.backward()
            self.qnet_optimizer1.step()
            
            self.qnet_optimizer2.zero_grad()
            loss_q2.backward()
            self.qnet_optimizer2.step()
            
            loss_q = torch.min(loss_q1, loss_q2)
            
            # Actor
            self.actor_optimizer.zero_grad()
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                s = batch.trajectories[:, 0, :self.observation_dim]
                a = batch.trajectories[:, 0, self.observation_dim:]
                g = batch.goals[:,0]
                v = self.qnet.q_min(s, a, g)
                loss_a = self.criterion(self.actor(torch.cat([s,g,v], -1)), a)
                loss_a = loss_a / self.gradient_accumulate_every
                loss_a.backward()
            self.actor_optimizer.step()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
                self.qnet.target_update()
                
            if self.step % self.logfreq == 0:
                output = {"loss_q": loss_q,
                          "q_her": q_her.mean(),
                          "q_ner": q_ner.mean(),
                          "loss_a": loss_a}
            
            self.step += 1
            
            if self.step % 1000 == 0:
                print(f'Evaluation... {step}/{n_train_steps}')
                with torch.no_grad():
                    total_return = 0
                    for epi in range(10):
                        env = self.dataset.env
                        state = env.reset()
                        env.set_target()
                        goal = env.get_target()
                        epi_return = 0
                        at_goal = False
                        for _ in range(env.max_episode_steps):
                            at_goal = state[:self.goal_dim]
                            action = self.actor(torch.cat([to_torch(state), 
                                                           to_torch(goal), 
                                                           torch.ones((1,), device='cuda')], -1))
                            next_state, reward, done, info = env.step(to_np(action))
                            epi_return += reward
                            if done: break
                            state = next_state
                        total_return += epi_return
                        print('epi return: ', epi_return)
                        
class FetchBCCritic(object):
    def __init__(self,
                 dataset,
                 critic,
                 renderer,
                 device,
                 # ema
                 step_start_ema=1000,
                 ema_decay=0.995,
                 update_ema_every=10,
                 train_batch_size=32,
                 gradient_accumulate_every=5,
                 lr=3e-4,
                 logfreq=1000,
                 load_path='/ext_hdd/sykim/DC/logs',
                 wandb=False,
                 ):
        self.observation_dim = dataset.observation_dim
        self.action_dim = dataset.action_dim
        self.goal_dim = 3
        
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(self.observation_dim+self.goal_dim+1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_dim)
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()
        
        self.critic = critic
        self.qnet = SimpleCritic(self.observation_dim, self.action_dim, self.goal_dim, dataset.normalizer)
        self.qnet_optimizer1 = torch.optim.Adam(self.qnet.qf1.parameters(), lr=lr)
        self.qnet_optimizer2 = torch.optim.Adam(self.qnet.qf2.parameters(), lr=lr)
        
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.dataset = dataset
        self.horizon = dataset.horizon
        self.dataloader_train = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, shuffle=True, drop_last=True,
        ))

        self.batch_size = train_batch_size
        
        self.gradient_accumulate_every = gradient_accumulate_every
        self.renderer = renderer
        self.logfreq = logfreq
        self.wandb = wandb
        self.step = 0
        
        if device == 'cuda':
            self.actor.cuda()
            self.ema_model.cuda()
            self.qnet.cuda()
        
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
    
    def train(self, n_train_steps):
        for train_step in range(int(n_train_steps)):
            self.actor.train()
            
            # Critic
            batch_her = next(self.dataloader_train)
            batch_her = batch_to_device(batch_her)
            batch_ner = next(self.dataloader_train)
            batch_ner = batch_to_device(batch_ner)
            if self.critic == 'HER':
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=batch_her.trajectories, 
                                          goal=batch_her.trajectories[:, 1, :self.goal_dim], 
                                          tdtarget=torch.ones((self.batch_size, 1), device='cuda'), 
                                          actor=self.actor)
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_ner.goals[:, 0],
                                          tdtarget=batch_ner.rewards[:, 0],
                                          actor=self.actor)
            elif self.critic == 'AM':
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=batch_her.trajectories,
                                          goal=batch_her.trajectories[:, 1, :self.goal_dim],
                                          tdtarget=torch.ones((self.batch_size, 1), device='cuda'),
                                          actor=self.actor)
                value = self.qnet.q_min(batch_ner.trajectories[:, 1, :self.observation_dim], 
                                        batch_ner.trajectories[:, 1, self.observation_dim:],
                                        batch_her.goals[:, 0])
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_her.goals[:, 0],
                                          tdtarget=self.qnet.gamma * value,
                                          actor=self.actor)
            elif self.critic == 'SSD':
                k = torch.randint(2, self.horizon+1, (self.batch_size,), device='cuda')
                trajectory_temp = torch.zeros_like(batch_her.trajectories[:,:2])
                for i, j in enumerate(k):
                    trajectory_temp[i] = batch_her.trajectories[i, self.horizon-j:self.horizon-j+2]
                loss1_her, loss2_her, q_her = self.qnet.loss(trajectory=trajectory_temp,
                                          goal=batch_her.trajectories[:, -1, :self.goal_dim],
                                          tdtarget=self.qnet.gamma**(k-2) * torch.ones((self.batch_size, 1), device='cuda'),
                                          actor=self.actor)
                value = self.qnet.q_min(batch_ner.trajectories[:, 1, :self.observation_dim], 
                                        batch_ner.trajectories[:, 1, self.observation_dim:],
                                        batch_her.goals[:, 0])
                loss1_ner, loss2_ner, q_ner = self.qnet.loss(trajectory=batch_ner.trajectories,
                                          goal=batch_her.goals[:, 0],
                                          tdtarget=self.qnet.gamma * value,
                                          actor=self.actor)
            loss_q1, loss_q2 = 0.5*(loss1_her+loss1_ner), 0.5*(loss2_her+loss2_ner)
            
            self.qnet_optimizer1.zero_grad()
            loss_q1.backward()
            self.qnet_optimizer1.step()
            
            self.qnet_optimizer2.zero_grad()
            loss_q2.backward()
            self.qnet_optimizer2.step()
            
            loss_q = torch.min(loss_q1, loss_q2)
            
            # Actor
            self.actor_optimizer.zero_grad()
            for _ in range(self.gradient_accumulate_every):
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                s = batch.trajectories[:, 0, :self.observation_dim]
                a = batch.trajectories[:, 0, self.observation_dim:]
                g = batch.goals[:,0]
                v = self.qnet.q_min(s, a, g)
                loss_a = self.criterion(self.actor(torch.cat([s,g,v], -1)), a)
                loss_a = loss_a / self.gradient_accumulate_every
                loss_a.backward()
            self.actor_optimizer.step()
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()
                self.qnet.target_update()
                
            if self.step % self.logfreq == 0:
                output = {"loss_q": loss_q,
                          "q_her": q_her.mean(),
                          "q_ner": q_ner.mean(),
                          "loss_a": loss_a}
            
            self.step += 1
            
            if self.step % 1000 == 0:
                print(f'Evaluation... {train_step}/{n_train_steps}')
                with torch.no_grad():
                    total_rewards = []
                    total_success_rate = []
                    total_ag, total_g = [], []
                    total_1st_step = []
                    for epi in range(10):
                        env = self.dataset.env
                        state = env.reset()
                        target = env.goal
                        at_goal = False
                        step, achieved_goal = 0, False
                        per_pos, per_rewards, per_success_rate = [], [], []
                        for _ in range(env.max_episode_steps):
                            at_goal = np.linalg.norm(state["achieved_goal"] - target) <= 0.01
                            action = self.actor(torch.cat([to_torch(state["observation"][np.arange(11)]), 
                                                           to_torch(target), 
                                                           torch.ones((1,), device='cuda')], -1))
                            next_state, reward, done, info = env.step(to_np(action))
                            step += 1
                            if 'score/success' in info:
                                info['is_success'] = float(info['score/success'])
                            if info['is_success'] and not achieved_goal:
                                achieved_goal = step
                            per_rewards.append(reward)
                            per_success_rate.append(info['is_success'])
                            if done: break
                            state = next_state
                        if not achieved_goal: achieved_goal=env.max_episode_steps
                        total_1st_step.append(achieved_goal)
                        total_ag.append(state["achieved_goal"])
                        total_g.append(target)
                        total_rewards.append(per_rewards)
                        total_success_rate.append(per_success_rate)
                        
                    total_g = np.array(total_g)
                    total_ag = np.array(total_ag)
                    distances = np.mean(np.linalg.norm(total_ag - total_g, axis=1))
                    dis_return, undis_return = discounted_return(np.array(total_rewards), gamma=0.98)
                    dis_return = np.mean(dis_return)
                    undis_return = np.mean(undis_return)
                    success_rate = np.array(total_success_rate)[:, -1].mean()

                    print(f'Discounted_Return: {dis_return:.2f} | Success_Rate: {success_rate:.2f} | Final_Distance: {distances:.2f}' )
                    print(total_1st_step)
  