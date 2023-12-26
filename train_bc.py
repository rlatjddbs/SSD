import utils
import os
import wandb
import argparse

from bc.agents import BCCritic, BC, FetchBCCritic
from datasets.sequence import SequenceDataset

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env', type=str, default='FetchPush-v1', help='the environment name')
    parser.add_argument('--critic', type=str, default='AM', help='her/am')
    parser.add_argument('--epoch', type=int, default=20, help='num epoch')
    parser.add_argument('--horizon', default=64, type=int)
    parser.add_argument('--n-train-steps', type=int, default=1e5, help='train steps per epoch')
    parser.add_argument('--max-path-length', type=int, default=600)
    parser.add_argument('--max-n-episodes', type=int, default=40000)
    parser.add_argument('--termination-penalty', type=float, default=None)
    parser.add_argument('--normalizer', type=str, default='LimitsNormalizer')
    parser.add_argument('--preprocess-fn', type=list, default=['fetch_dataset'])
    parser.add_argument('--seed', type=int, default=0, help='random seeds')
    parser.add_argument('--wandb', type=bool, default=True, help='random seeds')
    
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='the learning rate')
    parser.add_argument('--device', default='cuda', type=str, help='device')

    parser.add_argument('--expert_percent', type=float, default=0.1, help='the expert coefficient')
    parser.add_argument('--random_percent', type=float, default=0.9, help='the random coefficient')

    args = parser.parse_args()

    return args

args = get_args()
goal_dim = 2
renderer = utils.Maze2dRenderer(env='maze2d-umaze-v1')
dataset = SequenceDataset(env=args.env,
                          horizon=args.horizon,
                          normalizer=args.normalizer,
                          preprocess_fns=args.preprocess_fn,
                          max_path_length=args.max_path_length,
                          max_n_episodes=args.max_n_episodes,
                          termination_penalty=args.termination_penalty,
                          seed=args.seed
                          )

if args.critic == 'None':
    bc = BC(dataset=dataset,
            critic=args.critic,
            renderer=renderer,
            train_batch_size=args.batch_size,
            device=args.device,
            wandb=args.wandb
            )
else:
    if "Fetch" in dataset.env.name:
        bc = FetchBCCritic(
            dataset=dataset,
            critic=args.critic,
            renderer=renderer,
            train_batch_size=args.batch_size,
            device=args.device,
            wandb=args.wandb
        )
    elif "maze" in dataset.env.name:
        bc = BCCritic(dataset=dataset,
                critic=args.critic,
                renderer=renderer,
                train_batch_size=args.batch_size,
                device=args.device,
                wandb=args.wandb
                )



for i in range(args.epoch):
    print(f'Epoch {i} / {args.epoch}')
    bc.train(args.n_train_steps)
