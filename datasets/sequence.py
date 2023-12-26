from collections import namedtuple
import numpy as np
import torch
import pickle
import collections
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from d4rl import kitchen

Batch = namedtuple('Batch', 'trajectories rewards goals')
ValueBatch = namedtuple('ValueBatch', 'trajectories rtgs values')

def fetch_sequence_dataset(env, preprocess_fn):
    name = str.split(env.name, '-')[0]
    with open(f'../offline_gcrl_data/offline_data/mixed/{name}/buffer.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
        
class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=1,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.seed = seed
        
        if 'Fetch' in env.name:
            itr = fetch_sequence_dataset(env, self.preprocess_fn)
        else:
            itr = sequence_dataset(env, self.preprocess_fn)

        # fields = HERReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
            
        self.normalize()
        # if 'Fetch' in env.name or 'maze' in env.name:
        #     self.normalize(['achieved_goals'])

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'next_observations', 'rewards', 'goals']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        # next_observations = self.fields.normed_next_observations[path_ind, start:end]
        terminals = self.fields.terminals[path_ind, start:end]

        # conditions = self.get_conditions(observations)
        if hasattr(self.env, "_target") or hasattr(self.env, 'goal'):
            rewards = self.fields.rewards[path_ind, start:end]
            goals = self.fields.normed_goals[path_ind, start:end]
        else:
            rewards = self.fields.normed_rewards[path_ind, start:end]
            goals = self.fields.normed_rtgs[path_ind, start]
        
        # rtgs = self.fields.normed_rtgs[path_ind, start:end]
        # trajectories = np.concatenate([observations, actions, next_observations, rewards, terminals], axis=-1)
        # trajectories = np.concatenate([observations, actions, rewards, terminals], axis=-1)
        trajectories = np.concatenate([observations, actions], axis=-1)
        batch = Batch(trajectories, rewards, goals)
        return batch
