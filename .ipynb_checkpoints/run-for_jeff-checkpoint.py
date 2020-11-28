from datetime import datetime
import train
from ray import tune
import sys
import glob
import argparse
import torch
import numpy as np

def trainable(config, checkpoint_dir=None):
    experiment = train.Experiment(**config)
#     if config['checkpoint']:
#         experiment.load(config['checkpoint'])
    experiment.train()
    
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'env_id',
        metavar='env_id',
        type=str,
        help='OpenAI gym-formatted environment id.'
    )
    parser.add_argument(
        'name',
        type=str,
        help='Name for logs.'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to a checkpoint file to be used for warm starting or restarting.'
    )
    parser.add_argument(
        '--from_images',
        action='store_true',
        default=False,
        help='Whether or not to use image observations.'
    )
    parser.add_argument(
        '--fix_goals',
        action='store_true',
        default=False,
        help='Whether or not to keep the goal the same for the entire training and evaluation process.'
    )
    parser.add_argument(
        '--seed',
        default=1,
        help='Seed for reproducibility.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Torch device: one of "cuda" or "cpu".'
    )
    parser.add_argument(
        '--num_resampled_goals',
        type=int,
        default=1,
        help='Number of resampled goals for HER.'
    )
    parser.add_argument(
        '--capacity',
        type=int,
        default=1_000_000,
        help='Replay buffer capacity.'
    )
    parser.add_argument(
        '--feature_dim',
        type=int,
        default=128,
        help='Dimensionality of latent space output by convolutional encoder. Best values are between 64 and 256.'
    )
    parser.add_argument(
        '--hidden_sizes',
        type=list,
        default=[512, 512, 512],
        help='List of integers for sizes of hidden layers of the policy and critic networks.'
    )
    parser.add_argument(
        '--log_std',
        type=list,
        default=[-10, 2],
        help='A list of length two specifying the lower and upper log standard deviation range.'
    )
    parser.add_argument(
        '--discount',
        type=float,
        default=0.95,
        help='Discount factor (gamma) for discounting the future rewards.'
    )
    parser.add_argument(
        '--init_temperature',
        type=float,
        default=0.1,
        help='Initial temperature factor for entropy-regularized RL.'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0006,
        help='Learning rate for actor and critic.'
    )
    parser.add_argument(
        '--actor_update_frequency',
        type=int,
        default=2,
        help='How often to update the actor (policy) network relative to critic network updates.'
    )
    parser.add_argument(
        '--critic_tau',
        type=float,
        default=0.005,
        help='The weight used for Polyak averaging (exponential smoothing) of the critic target network updates.'
    )
    parser.add_argument(
        '--critic_target_update_frequency',
        type=int,
        default=2,
        help='How often to update the target critic network relative to based critic network updates.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='The number of samples drawn by the actor and critic for updating. Includes HER samples.'
    )
    parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=5,
        help='The number of evaluation episodes used on the agent. More gives a lower variance estimate of performance.'
    )
    parser.add_argument(
        '--gradient_steps',
        type=int,
        default=1,
        help='The number of gradient steps per environment step.'
    )
    parser.add_argument(
        '--num_timesteps',
        type=int,
        default=20_000,
        help='The total number of environment interaction steps (excluding evaluation). Includes steps taken before training.'
    )
    parser.add_argument(
        '--num_seed_steps',
        type=int,
        default=1_000,
        help='A uniform random policy will be executed for this many time steps at the beginning of training to help diversify the replay buffer.'
    )
    parser.add_argument(
        '--update_after',
        type=int,
        default=1_000,
        help='When to begin doing gradient descent. This must be larger than batch_size. This can be smaller than num_seed_steps, but it generally makes sense to make this equal to num_seed_steps'
    )
    parser.add_argument(
        '--eval_every',
        type=int,
        default=20,
        help='How often to do evaluation, in the number of episodes.'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=5_000,
        help='How often to save the experiment state, in the number of time steps.'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of samples for ray tune to use.'
    )
    parser.add_argument(
        '--cpu',
        type=int,
        default=1,
        help='Number of cpus per trial for ray tune to use.'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=1 if torch.cuda.is_available() else 0,
        help='Number of gpus per trial for ray tune to use.'
    )
    
    args = parser.parse_args()
    # convert args namespace to a dictionary
    # replace the values of the dictionary like value -> tune.grid_search([value, ])
    # so that ray tune logs them.
    args_config = {param: tune.grid_search([value, ]) for param, value in vars(args).items()}
    
    config = {
        'env_id':tune.grid_search(['FetchPickAndPlace-v1']),
        'fix_goals':tune.grid_search([False,]),
        'num_resampled_goals':tune.grid_search([1]),
        'hidden_sizes':tune.grid_search([
            [512, 512],
        ]),
        'discount':tune.grid_search([
            0.95,
        ]),
        'init_temperature':tune.grid_search([
            1.,
        ]),
        'lr':tune.grid_search([0.0005, ]),
        'actor_update_frequency':tune.grid_search([1,]),
        'critic_tau':tune.grid_search([0.0005, ]),
        'batch_size':tune.grid_search([1024]),
        'gradient_steps':tune.grid_search([1,]),
        'update_after':tune.grid_search([10_000]),
        'num_seed_steps':tune.grid_search([10_000]),
        'eval_every':tune.grid_search([100]),
        'num_timesteps':tune.grid_search([4_000_000]),
        
        'seed':tune.grid_search(list(range(8))),

        'save_every':np.inf
    }
    
    tune.run(
        trainable,
        name=args.name,
        config=config,
#         num_samples=args.num_samples,
        num_samples=1,
        resources_per_trial={
#             'cpu':args.cpu, 
#             'gpu':args.gpu,
            'cpu':1, 
            'gpu':0.5,
        },
        local_dir=f'logs',
        log_to_file=True,
)