import gym
import numpy as np
import einops
from scipy.spatial.transform import Rotation as R
import pdb

from .d4rl import load_environment

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#
def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

def discount_cumsum(x, gamma=1.):
    discount_cumsum = np.zeros_like(x)
    if len(x) == 0: return discount_cumsum
    discount_cumsum[...,-1] = x[...,-1]
    for t in reversed(range(x.shape[-1]-1)):
        discount_cumsum[...,t] = x[...,t] + gamma * discount_cumsum[...,t+1]
    return discount_cumsum

def compose(*fns):

    def _fn(x):
        for fn in fns:
            x = fn(x)
        return x

    return _fn

def get_preprocess_fn(fn_names, env):
    fns = [eval(name)(env) for name in fn_names]
    return compose(*fns)

def get_policy_preprocess_fn(fn_names):
    fns = [eval(name) for name in fn_names]
    return compose(*fns)

#-----------------------------------------------------------------------------#
#-------------------------- preprocessing functions --------------------------#
#-----------------------------------------------------------------------------#

#------------------------ @TODO: remove some of these ------------------------#

def arctanh_actions(*args, **kwargs):
    epsilon = 1e-4

    def _fn(dataset):
        actions = dataset['actions']
        assert actions.min() >= -1 and actions.max() <= 1, \
            f'applying arctanh to actions in range [{actions.min()}, {actions.max()}]'
        actions = np.clip(actions, -1 + epsilon, 1 - epsilon)
        dataset['actions'] = np.arctanh(actions)
        return dataset

    return _fn

def add_deltas(env):

    def _fn(dataset):
        deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn

def mujoco_set_goals(env):
    env = load_environment(env) if type(env) == str else env
    def _fn(dataset):
        dataset['rtgs'] = np.zeros_like(dataset['rewards'])
        dataset['goals'] = np.zeros_like(dataset['rewards'])
        start = 0
        for i in range(len(dataset['observations'])):
            if dataset['timeouts'][i] or dataset['terminals'][i]:
                rewards = dataset['rewards'][start:i]
                rtg = discount_cumsum(rewards)
                dataset['rtgs'][start:i] = rtg
                dataset['goals'][start:i] = rtg
                start = i
        rewards = dataset['rewards'][start:]
        rtg = discount_cumsum(rewards)
        dataset['rtgs'][start:] = rtg
        dataset['goals'][start:] = rtg
        return dataset
    return _fn
        

def maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    goal = np.array(env._target)
    threshold = 0.5

    def _fn(dataset):
        dataset['next_observations'] = np.concatenate([dataset['observations'][1:], dataset['observations'][-1,None]], 0)
        dataset['rtgs'] = np.zeros_like(dataset['rewards'])
        start = 0
        for i in range(len(dataset['observations'])):
            if dataset['timeouts'][i] or dataset['terminals'][i]:
                rewards = dataset['rewards'][start:i]
                rtg = discount_cumsum(rewards)
                dataset['rtgs'][start:i] = rtg
                start = i
        rewards = dataset['rewards'][start:]
        rtg = discount_cumsum(rewards)
        dataset['rtgs'][start:] = rtg
        
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])
        goals = np.zeros_like(dataset['infos/goal'])
        goals[:] = goal

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        dataset['goals'] = dataset['infos/goal']
        return dataset

    return _fn


#---------------------- Kitchen ------------------------#

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.3

def kitchen_dataset(env):
    env = load_environment(env) if type(env) == str else env
    threshold = 0.3

    def check_tasks_completed(observations):
        '''
            observations: [N x 60]
        '''
        N = len(observations)
        obj_qp = observations[:, 9:30]
        goal = observations[:, 30:]
        
        tasks_completed = [[].copy() for _ in range(N)]
        for element in env.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = ((obj_qp[..., element_idx-9] - goal[..., element_idx]) ** 2).sum(axis=1) ** 0.5
            for i in np.where(distance < threshold)[0]:
                tasks_completed[i].append(element)
                
        return tasks_completed
        
    
    def _fn(dataset):
        # For path lengths
        N = len(dataset['rewards'])
        subends = np.where(dataset['rewards'][1:] - dataset['rewards'][:-1] != 0)[0]
        subpath_lengths = np.concatenate([subends, [N-1]]) - np.concatenate([[-1], subends])
        n, l = len(subpath_lengths), subpath_lengths.max(), 
        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(subpath_lengths)} paths | '
            f'min length: {subpath_lengths.min()} | max length: {subpath_lengths.max()}'
        )
        
        # For next_observations
        observations_split = np.split(dataset['observations'][..., :30], subends)
        for i, obs in enumerate(observations_split):
            if len(obs) > 1:
                observations_split[i] = np.pad(obs[1:], mode='edge', pad_width=((0,1), (0,0))).copy()
            else:
                observations_split[i] = obs
                
        # For timeouts which means endpoint of each subpath.
        timeouts = np.zeros_like(dataset['timeouts'])
        timeouts[subends] = 1
        
        # For subgoals which means goal of each timestep, among sequence of the goals. 
        tasks_completed = check_tasks_completed(dataset['observations'])
        subgoals = np.zeros_like(dataset['observations'][..., :30])
        subepi_start = 0
        for i in np.where(timeouts)[0] + 1:
            tasks_after = tasks_completed[i]
            tasks_before = tasks_completed[i-1]
            tasks_added = list(set(tasks_after) - set(tasks_before))
            tasks_remain = set(env.tasks_to_complete) - set(tasks_after)
            if len(tasks_added) > 0:
                subgoals[subepi_start:i, OBS_ELEMENT_INDICES[tasks_added[0]]] = OBS_ELEMENT_GOALS[tasks_added[0]]
            else:
                for task in tasks_remain:
                    subgoals[subepi_start:i, OBS_ELEMENT_INDICES[task]] += OBS_ELEMENT_GOALS[task]
            subepi_start = i
        if len(tasks_added) > 0:
            subgoals[subepi_start:, OBS_ELEMENT_INDICES[tasks_added[0]]] = OBS_ELEMENT_GOALS[tasks_added[0]]
        else:
            for task in tasks_remain:
                subgoals[subepi_start:, OBS_ELEMENT_INDICES[task]] += OBS_ELEMENT_GOALS[task]
        
        
        dataset_new = {'observations': dataset['observations'][..., :30].copy(), 
                       'next_observations': np.concatenate(observations_split, axis=0),
                       'actions': dataset['actions'].copy(), 
                       'rewards': dataset['rewards'].copy(), 
                       'goals': subgoals,
                       'achieved_goals': dataset['observations'][..., :30].copy(),
                       'terminals': dataset['terminals'].copy(),
                       'timeouts': timeouts}  
        
        return dataset_new
        
    return _fn
        

def her_maze2d_set_terminals(env):
    env = load_environment(env) if type(env) == str else env
    threshold = 0.5

    def _fn(dataset):
        # dataset['next_observations'] = np.concatenate([dataset['observations'][1:], dataset['observations'][-1,None]], 0)
        dataset['rtgs'] = np.zeros_like(dataset['rewards'])
        start = 0
        for i in range(len(dataset['observations'])):
            if dataset['timeouts'][i] or dataset['terminals'][i]:
                rewards = dataset['rewards'][start:i]
                rtg = discount_cumsum(rewards)
                dataset['rtgs'][start:i] = rtg
                start = i
        rewards = dataset['rewards'][start:]
        rtg = discount_cumsum(rewards)
        dataset['rtgs'][start:] = rtg
            
        her_goal = np.zeros_like(dataset['infos/goal'])
        start = 0
        for end in np.where(dataset['timeouts'])[0]+1:
            her_goal[start:end] = dataset['observations'][end-1, :2]
            start = end
        her_goal[start:] = dataset['observations'][-1, :2]
        xy = dataset['observations'][:,:2]
        distances = np.linalg.norm(xy - her_goal, axis=-1)
        at_goal = distances < threshold
        timeouts = np.zeros_like(dataset['timeouts'])

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        timeouts[:-1] = at_goal[:-1] * ~at_goal[1:]
        # rewards = at_goal.astype(np.float32)

        timeout_steps = np.where(timeouts)[0]
        path_lengths = timeout_steps[1:] - timeout_steps[:-1]

        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {len(path_lengths)} paths | '
            f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
        )

        dataset['timeouts'] = timeouts
        dataset['goals'] = her_goal
        dataset['achieved_goals'] = dataset['observations'][...,:2]
        dataset['next_observations'] = np.concatenate([dataset['observations'][1:], dataset['observations'][-1,None]], 0)
        # dataset['rewards'] = rewards
        return dataset

    return _fn

def fetch_dataset(env):
    env = load_environment(env) if type(env) == str else env
    threshold = 0.05
    
    def _fn(dataset):
        xyz = dataset['ag'][:,:-1]
        distances = np.linalg.norm(xyz-dataset['g'], axis=-1)
        at_goal = distances < threshold
        shape = dataset['u'].shape[:-1]
        timeouts = np.zeros(shape)
        
        next_xyz = dataset['ag'][:,1:]
        next_distances = np.linalg.norm(next_xyz-dataset['g'], axis=-1)
        rewards = (next_distances < threshold).astype(np.float32)

        ## timeout at time t iff
        ##      at goal at time t and
        ##      not at goal at time t + 1
        # timeouts[:,:-1] = at_goal[:, :-1] * ~at_goal[:, 1:]
        # if env.reward_type == 'sparse':
        #     rewards = -(~at_goal).astype(np.float32)
        # elif env.reward_type == 'very_sparse':
        #     rewards = at_goal.astype(np.float32)
        # else:
        #     rewards = -distances
        
        timeouts[:, -1] = 1
        # path_lengths = []
        # for i in range(shape[0]):
        #     path_lengths = np.concatenate([path_lengths, np.where(timeouts[i])[0][1:] - np.where(timeouts[i])[0][:-1]], axis=0)
        
        print(
            f'[ utils/preprocessing ] Segmented {env.name} | {shape[0]} paths | '
            f'min length: {shape[-1]} | max length: {shape[-1]}'
        )

        rtg = discount_cumsum(rewards)
        dataset['rtgs'] = rtg.reshape((np.prod(shape), -1))
        if env.has_object:
            extract = np.arange(11)
            dataset['observations'] = dataset['o'][:,:-1, extract].reshape((np.prod(shape), -1))
            dataset['next_observations'] = dataset['o'][:,1:, extract].reshape((np.prod(shape), -1))
        else:
            dataset['observations'] = dataset['o'][:,:-1].reshape((np.prod(shape), -1))
            dataset['next_observations'] = dataset['o'][:,1:].reshape((np.prod(shape), -1))
        dataset['actions'] = dataset['u'].reshape((np.prod(shape), -1))
        dataset['rewards'] = rewards.reshape((np.prod(shape), -1))
        dataset['timeouts'] = timeouts.reshape((np.prod(shape), -1))
        dataset['goals'] = dataset['g'].reshape((np.prod(shape), -1))
        dataset['achieved_goals'] = dataset['ag'][:,:-1].reshape((np.prod(shape), -1))
        dataset['terminals'] = np.zeros_like(dataset['timeouts']).astype(np.bool8)
        del dataset['o']
        del dataset['u']
        del dataset['ag']
        del dataset['g']
        return dataset

    return _fn
"""#-------------------------- block-stacking --------------------------#

def blocks_quat_to_euler(observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1

        returns : [ N x robot_dim + n_blocks * 10] = [ N x 47 ]
            xyz: 3
            sin: 3
            cos: 3
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        quat = block_info[:, 3:-1]
        contact = block_info[:, -1:]

        euler = R.from_quat(quat).as_euler('xyz')
        sin = np.sin(euler)
        cos = np.cos(euler)

        X = np.concatenate([
            X,
            xpos,
            sin,
            cos,
            contact,
        ], axis=-1)

    return X

def blocks_euler_to_quat_2d(observations):
    robot_dim = 7
    block_dim = 10
    n_blocks = 4

    assert observations.shape[-1] == robot_dim + n_blocks * block_dim

    X = observations[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]

        xpos = block_info[:, :3]
        sin = block_info[:, 3:6]
        cos = block_info[:, 6:9]
        contact = block_info[:, 9:]

        euler = np.arctan2(sin, cos)
        quat = R.from_euler('xyz', euler, degrees=False).as_quat()

        X = np.concatenate([
            X,
            xpos,
            quat,
            contact,
        ], axis=-1)

    return X

def blocks_euler_to_quat(paths):
    return np.stack([
        blocks_euler_to_quat_2d(path)
        for path in paths
    ], axis=0)

def blocks_process_cubes(env):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = blocks_quat_to_euler(dataset[key])
        return dataset

    return _fn

def blocks_remove_kuka(env):

    def _fn(dataset):
        for key in ['observations', 'next_observations']:
            dataset[key] = dataset[key][:, 7:]
        return dataset

    return _fn

def blocks_add_kuka(observations):
    '''
        observations : [ batch_size x horizon x 32 ]
    '''
    robot_dim = 7
    batch_size, horizon, _ = observations.shape
    observations = np.concatenate([
        np.zeros((batch_size, horizon, 7)),
        observations,
    ], axis=-1)
    return observations

def blocks_cumsum_quat(deltas):
    '''
        deltas : [ batch_size x horizon x transition_dim ]
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert deltas.shape[-1] == robot_dim + n_blocks * block_dim

    batch_size, horizon, _ = deltas.shape

    cumsum = deltas.cumsum(axis=1)
    for i in range(n_blocks):
        start = robot_dim + i * block_dim + 3
        end = start + 4

        quat = deltas[:, :, start:end].copy()

        quat = einops.rearrange(quat, 'b h q -> (b h) q')
        euler = R.from_quat(quat).as_euler('xyz')
        euler = einops.rearrange(euler, '(b h) e -> b h e', b=batch_size)
        cumsum_euler = euler.cumsum(axis=1)

        cumsum_euler = einops.rearrange(cumsum_euler, 'b h e -> (b h) e')
        cumsum_quat = R.from_euler('xyz', cumsum_euler).as_quat()
        cumsum_quat = einops.rearrange(cumsum_quat, '(b h) q -> b h q', b=batch_size)

        cumsum[:, :, start:end] = cumsum_quat.copy()

    return cumsum

def blocks_delta_quat_helper(observations, next_observations):
    '''
        input : [ N x robot_dim + n_blocks * 8 ] = [ N x 39 ]
            xyz: 3
            quat: 4
            contact: 1
    '''
    robot_dim = 7
    block_dim = 8
    n_blocks = 4
    assert observations.shape[-1] == next_observations.shape[-1] == robot_dim + n_blocks * block_dim

    deltas = (next_observations - observations)[:, :robot_dim]

    for i in range(n_blocks):
        start = robot_dim + i * block_dim
        end = start + block_dim

        block_info = observations[:, start:end]
        next_block_info = next_observations[:, start:end]

        xpos = block_info[:, :3]
        next_xpos = next_block_info[:, :3]

        quat = block_info[:, 3:-1]
        next_quat = next_block_info[:, 3:-1]

        contact = block_info[:, -1:]
        next_contact = next_block_info[:, -1:]

        delta_xpos = next_xpos - xpos
        delta_contact = next_contact - contact

        rot = R.from_quat(quat)
        next_rot = R.from_quat(next_quat)

        delta_quat = (next_rot * rot.inv()).as_quat()
        w = delta_quat[:, -1:]

        ## make w positive to avoid [0, 0, 0, -1]
        delta_quat = delta_quat * np.sign(w)

        ## apply rot then delta to ensure we end at next_rot
        ## delta * rot = next_rot * rot' * rot = next_rot
        next_euler = next_rot.as_euler('xyz')
        next_euler_check = (R.from_quat(delta_quat) * rot).as_euler('xyz')
        assert np.allclose(next_euler, next_euler_check)

        deltas = np.concatenate([
            deltas,
            delta_xpos,
            delta_quat,
            delta_contact,
        ], axis=-1)

    return deltas

def blocks_add_deltas(env):

    def _fn(dataset):
        deltas = blocks_delta_quat_helper(dataset['observations'], dataset['next_observations'])
        # deltas = dataset['next_observations'] - dataset['observations']
        dataset['deltas'] = deltas
        return dataset

    return _fn"""