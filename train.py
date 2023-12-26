import argparse
import gym
import numpy as np
import os
import wandb

import utils
from datasets import SequenceDataset
from dc.dc import DiffuserCritic
from dc.dd import DecisionDiffuser

class IterParser(utils.HparamEnv):
    dataset: str = 'FetchReach-v1'
    config: str = 'config.fetch'
    experiment: str = 'diffusion'

iterparser = IterParser()

class Parser(utils.Parser):
    pid: int = 0
    cid: float = 0

args = Parser().parse_args(iterparser)

if 'maze2d' in args.dataset:
    goal_dim = 2
    renderer = utils.Maze2dRenderer(env=args.dataset)
elif 'Fetch' in args.dataset:
    goal_dim = 3
    renderer = utils.FetchRenderer(env=args.dataset)
else:
    goal_dim = 1
    renderer = utils.MuJoCoRenderer(env=args.dataset)

dataset = SequenceDataset(
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    max_path_length=args.max_path_length,
    max_n_episodes=args.max_n_episodes,
    use_padding=args.use_padding,
    termination_penalty=args.termination_penalty,
    seed=args.seed,
)

dc = DiffuserCritic(
    dataset=dataset,
    renderer=renderer,
    goal_dim=goal_dim,
    device=args.device,
    dim_mults=args.dim_mults,
    conditional=args.conditional,
    condition_dropout=args.condition_dropout,
    calc_energy=args.calc_energy,
    n_timesteps=args.n_diffusion_steps,
    clip_denoised=args.clip_denoised,
    condition_guidance_w=args.condition_guidance_w,
    beta_schedule=args.beta_schedule,
    action_weight=args.action_weight,
    # warmup_steps=args.warmup_steps,
    maxq=args.maxq,
    alpha=args.alpha, 
    ema_decay=args.ema_decay,
    train_batch_size=args.batch_size,
    gradient_accumulate_every=args.gradient_accumulate_every,
    lr=args.lr,
    logdir=f'{args.logbase}/{args.dataset}/{args.exp_name}',
    log_freq=args.log_freq,
    save_freq=int(args.n_train_steps // args.n_saves),
    label_freq=int(args.n_train_steps // args.n_saves),
    wandb=args.wandb,
)
# dc = DecisionDiffuser(
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
#     log_freq=args.log_freq,
#     save_freq=int(args.n_train_steps // args.n_saves),
#     label_freq=int(args.n_train_steps // args.n_saves),
#     wandb=args.wandb,
# )
    
utils.report_parameters(dc.diffuser)
utils.report_parameters(dc.critic)
# utils.setup_dist()

# print('Testing forward...', end=' ', flush=True)
# batch = utils.batchify(dataset[0])
# loss = dc.diffuser.loss(*batch)
# loss.backward()
# print('✓')

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    dc.train(n_train_steps=args.n_steps_per_epoch)
    
