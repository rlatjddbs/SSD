import json
import wandb
import torch
import os
import numpy as np

from utils.helpers import discounted_return
# from d4rl.kitchen.kitchen_envs import OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS

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
##############################################################################
################################ Config setup ################################
##############################################################################

# class IterParser(utils.HparamEnv):
#     dataset: str = 'FetchPickAndPlace-v1'
#     config: str = 'config.fetch'
#     experiment: str = 'evaluate'

# iterparser = IterParser()

# class Parser(utils.Parser):
#     pid: int = 0
#     cid: float = 0

# args = Parser().parse_args(iterparser)

##############################################################################
################################### Setup ####################################
##############################################################################

# env = datasets.load_environment(args.dataset)
# # env = wrappers.Monitor(env, f'{args.logbase}/{args.dataset}/{args.exp_name}', force=True)
# # env.seed(args.epi_seed)
# horizon = args.horizon

# dataset = datasets.SequenceDataset(
#     env=args.dataset,
#     horizon=args.horizon,
#     normalizer=args.normalizer,
#     preprocess_fns=args.preprocess_fns,
#     max_path_length=args.max_path_length,
#     max_n_episodes=args.max_n_episodes,
#     use_padding=args.use_padding,
#     seed=args.seed,
# )

# observation_dim = dataset.observation_dim
# action_dim = dataset.action_dim

# if 'maze2d' in args.dataset:
#     goal_dim = 2
#     renderer = utils.Maze2dRenderer(env=args.dataset)
# elif 'Fetch' in args.dataset:
#     goal_dim = 3
#     renderer = utils.FetchRenderer(env=args.dataset)
# else:
#     goal_dim = 1
#     renderer = utils.MuJoCoRenderer(env=args.dataset)


# dc = DiffuserCritic(
#     dataset=dataset,
#     renderer=renderer,
#     goal_dim=goal_dim,
#     device=args.device,
#     dim_mults=args.dim_mults,
#     conditional=args.conditional,
#     condition_dropout=args.condition_dropout,
#     calc_energy=args.calc_energy,
#     n_timesteps=args.n_diffusion_steps,
#     clip_denoised=args.clip_denoised,
#     condition_guidance_w=args.condition_guidance_w,
#     beta_schedule=args.beta_schedule,
#     action_weight=args.action_weight,
#     # warmup_steps=args.warmup_steps,
#     maxq=args.maxq,
#     alpha=args.alpha, 
#     ema_decay=args.ema_decay,
#     train_batch_size=args.batch_size,
#     gradient_accumulate_every=args.gradient_accumulate_every,
#     lr=args.lr,
#     logdir=f'{args.logbase}/{args.dataset}/{args.exp_name}',
#     diffusion_loadpath=f'{args.logbase}/{args.dataset}/{args.diffusion_loadpath}',
#     log_freq=args.log_freq,
#     save_freq=int(args.n_train_steps // args.n_saves),
#     label_freq=int(args.n_train_steps // args.n_saves),
#     wandb=args.wandb,
# )

# dc.load(args.diffusion_epoch)

# try: 
#     has_object = dataset.env.has_object
# except:
#     has_object = False
    
# if args.control == 'torque':
#     policy = GoalTorqueControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
# elif args.control == 'position':
#     policy = GoalPositionControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
# elif args.control == 'every':
#     policy = SampleEveryControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
# elif args.control == 'fetch':
#     policy = FetchControl(dc.ema_model, dataset.normalizer, observation_dim, goal_dim, has_object)
# else: 
#     NotImplementedError(args.control)

# ## Init wandb
# if args.wandb:
#     print('Wandb init...')
#     wandb_dir = '/tmp/sykim/wandb'
#     os.makedirs(wandb_dir, exist_ok=True)
#     wandb.init(project=args.prefix.replace('/', '-'),
#                entity='sungyoon',
#                config=args,
#                dir=wandb_dir,
#                )
#     wandb.run.name = f"rand_{args.dataset}"

##############################################################################
############################## Start iteration ###############################
##############################################################################
# state = env.reset()

# ## Set target and condition
# if 'maze2d' in args.dataset:
#     if args.multi: 
#         print('Resetting target')
#         env.set_target()
#     ## set conditioning xy position to be the goal
#     target = env._target
# elif 'Fetch' in args.dataset:
#     ## set conditioning xyz position to be the goal
#     target = env.goal
# else:
#     ## set conditioning rtg to be the goal
#     target = reverse_normalized_score(args.dataset, args.target_rtg)
#     # target = dataset.normalizer(target, 'rtgs')
# condition = torch.ones((1, horizon, 1)).to(args.device)
# # condition[0, -1] = 1
# gamma = dc.critic.gamma

# total_reward = 0
# rollout = []
# rollout_sim = []
# rewards = []
# at_goal = False

def increasing_schedule(t, gamma, horizon, max_epi_len):
    return gamma ** (horizon * ((max_epi_len - t) / max_epi_len))
    # return (1 - gamma ** horizon) * (t / max_epi_len) + gamma ** horizon

def main(env, n_episodes, policy, target_v):
    succ_rates = []
    undisc_returns = []
    disc_returns = []
    distances = []
    for _ in range(n_episodes):
        total_reward = 0
        rewards = []
        at_goal = False
        state = env.reset()
        for t in range(env.max_episode_steps):

            at_goal = np.linalg.norm(state['achieved_goal'] - state['desired_goal']) <= 0.05
            if env.has_object:
                observation = state['observation'][np.arange(11)]
            else:
                observation = state['observation']

            # if args.increasing_condition:
            #     condition = torch.ones((1, horizon, 1)).to(args.device) * gamma ** (1 - ((t + horizon) / env.max_episode_steps))
            condition = torch.ones((1, 1)).to('cuda') * target_v
            action = policy.act(observation, condition, state['desired_goal'], at_goal)

            # # Store rollout for rendering
            # if 'Fetch' in env.name:
            #     rollout_sim.append(copy.deepcopy(env.sim.get_state()))
            # else:
            #     rollout.append(observation[None, ].copy())
            
            # Step
            next_state, reward, done, _ = env.step(action)
            
            reward += 1
            rewards.append(reward)
            dis_return, total_reward = discounted_return(np.array(rewards), 0.98)
            distance = np.linalg.norm(state['achieved_goal'] - state['desired_goal'])
            output = {'reward': reward, \
                    'total_reward': total_reward, \
                    'discounted_return': dis_return, \
                    'distance': distance}

            
            # output_str = ' | '.join([f'{k}: {v:.4f}' for k, v in output.items()])
            # print(
            #     f't: {t} | {output_str} |'
            #     f'{action}'
            # )
            
            if done:
                break
            state = next_state
        succ_rates.append(at_goal)
        undisc_returns.append(total_reward)
        disc_returns.append(dis_return)
        distances.append(distance)
    return succ_rates, undisc_returns, disc_returns, distances


def main_maze(env, n_episodes, policy, target_v):
    succ_rates = []
    undisc_returns = []
    scores = []
    distances = []
    for _ in range(n_episodes):
        total_reward = 0 
        rewards = []
        at_goal = False
        state = env.reset()
        target = env._target
        for t in range(env.max_episode_steps):
            at_goal = np.linalg.norm(state[:2] - target) <= 0.5
            
            condition = torch.ones((1, 1)).to('cuda') * target_v

            action = policy.act(state, condition, target, at_goal)
            
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            rewards.append(reward)
            score = env.get_normalized_score(total_reward)
            distance = np.linalg.norm(state[:2] - target)
            
            if done:
                break
            state = next_state
        succ_rates.append(at_goal)
        undisc_returns.append(total_reward)
        scores.append(score)
        distances.append(distance)
    return succ_rates, undisc_returns, scores, distances


def main_kitchen(env, n_episodes, policy, target_v):
    succ_rates = []
    undisc_returns = []
    disc_returns = []
    for _ in range(n_episodes):
        total_reward = 0
        rewards = []
        state = env.reset()
        at_goal = False
        tasks_to_complete = list(env.tasks_to_complete)
        for t in range(env.max_episode_steps):
            
            target = np.zeros_like(state[:30])
            for task in tasks_to_complete:
                target[OBS_ELEMENT_INDICES[task]] += OBS_ELEMENT_GOALS[task]
            at_goal = np.linalg.norm(state[:30] - target) <= 0.3
            observation = state[:30]

            # if args.increasing_condition:
            #     condition = torch.ones((1, horizon, 1)).to(args.device) * gamma ** (1 - ((t + horizon) / env.max_episode_steps))
            condition = torch.ones((1, 1)).to('cuda') * target_v
            action = policy.act(observation, condition, target, at_goal)
            
            # Step
            next_state, reward, done, _ = env.step(action)
            
            total_reward += reward
            rewards.append(reward)
            dis_return, total_reward = discounted_return(np.array(rewards), 0.98)
            output = {'reward': reward, \
                    'total_reward': total_reward, \
                    'discounted_return': dis_return}

            
            # output_str = ' | '.join([f'{k}: {v:.4f}' for k, v in output.items()])
            # print(
            #     f't: {t} | {output_str} |'
            #     f'{action}'
            # )
            
            if done:
                break
            state = next_state
        succ_rates.append(at_goal)
        undisc_returns.append(total_reward)
        disc_returns.append(dis_return)
    return succ_rates, undisc_returns, disc_returns, None

# Rendering
# if 'Fetch' in args.dataset:
#     success = (reward == 1)
#     print('success:', success)
#     renderer.composite(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.png', rollout_sim)
#     # env.close()
# else:
#     renderer.composite(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.png', rollout)
    
# renderer.render_rollout(f'{args.logbase}/{args.dataset}/{args.exp_name}/rollout.mp4', rollout_sim)

# if args.wandb:
#     wandb.finish()
