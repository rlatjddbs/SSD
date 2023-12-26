import socket
from utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('seed', 's'),
    # ('beta_schedule', 'beta'),
]


eval_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
    ('epi_seed', 's'),
]

base = {

    'diffusion': {
        ## model
        'horizon': [16], # 4n + 2
        'n_diffusion_steps': [50],
        'action_weight': [10],
        'loss_weights': [None],
        'loss_discount': [1],
        'predict_epsilon': [True],
        'calc_energy': [False],
        'dim_mults': [(1,4,8)],

        ## dataset
        'termination_penalty': [None],
        'normalizer': ['LimitsNormalizer'],
        'preprocess_fns': [['fetch_dataset']],
        'use_padding': [False],
        'max_path_length': [50],
        'max_n_episodes': [100000],
        
        ## diffuser
        'conditional': [True],
        'condition_guidance_w': [1.2],
        'condition_dropout': [0.25],
        'beta_schedule': ['cosine'],
        'clip_denoised': [True],

        ## serialization
        # 'logbase': ['/ext2/sykim/DC/logs'],
        'logbase': ['logs'],
        'prefix': ['dc/transformer3'],
        'exp_name': [watch(diffusion_args_to_watch)],

        ## training
        'seed': [5],
        'maxq': [False],
        'alpha': [1],
        'n_steps_per_epoch': [1000],
        'loss_type': ['l2'],
        'n_train_steps': [2e6],
        'warmup_steps': [4e5],
        'batch_size': [256],
        'lr': [2e-4],
        'gradient_accumulate_every': [2],
        'ema_decay': [0.995],
        # 'save_freq': [5000],
        'sample_freq': [5000],
        'log_freq': [10000],
        'n_saves': [5],
        'save_parallel': [False],
        'n_reference': [50],
        'n_samples': [10],
        'bucket': [None],
        'device': ['cuda'],
        'wandb': [False],
    },
    
    'evaluate': {
        # 'guide': 'sampling.ValueGuide',
        'target_v': [0.0, 0.8, 1.0, 1.2, 1.4],
        # 'decreasing_target_rtg': [True],
        # 'policy': ['sampling.DDPolicyV2'],
        # 'max_episode_length': [1000],
        'batch_size': [1],
        'preprocess_fns': [['fetch_dataset']],
        'device': ['cuda'],
        'epi_seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        'wandb': [False],

        ## sample_kwargs
        'n_guide_steps': [2],
        'scale': [0.1],
        't_stopgrad': [2],
        'scale_grad_by_std': [True],
        'n_initial_steps': [1],
        'update_policy_every': [2],
        'control': ['fetch'],
        'increasing_condition': [False],
        
        ## serialization
        'loadbase': [None],
        'logbase': ['./logs'],
        # 'logbase': ['/ext2/sykim/DC/logs'],
        'prefix': ['eval/final'],
        'exp_name': [watch(eval_args_to_watch)],
        'vis_freq': [10],
        'max_render': [8],

        ## diffusion model
        'horizon': [16], #None,
        'n_diffusion_steps': [50],
        'seed': [6, 1,2,3],

        ## loading
        'diffusion_loadpath': ['f:dc/transformer3_H{horizon}_T{n_diffusion_steps}_s{seed}'],
        'diffusion_epoch': [799999],

        'verbose': [False],
        'suffix': ['0'],
    },
    


}

#------------------------ overrides ------------------------#


# hopper_medium_expert_v2 = {
#     'plan': {
#         'scale': 0.001,
#         't_stopgrad': 4,
#     },
# }


# halfcheetah_medium_replay_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
#     'diffusion': {
#         'horizon': 1,
#         'dim_mults': (1, 4, 8),
#         'attention': True,
#     },
#     'values': {
#         'horizon': 4,
#         'dim_mults': (1, 4, 8),
#     },
#     'plan': {
#         'horizon': 4,
#         # 'n_guide_steps': 1,
#         'scale': 0.001,
#         't_stopgrad': 4,
#     },
# }
