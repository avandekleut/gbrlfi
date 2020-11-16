from datetime import datetime
import train
from ray import tune
import sys

def trainable(config, checkpoint_dir=None):
    train.Experiment(**config['experiment_config']).train(**config['train_config'])
    
if __name__ == "__main__": 
    config = dict(
        experiment_config=dict(
#             env_id=tune.grid_search(["FetchReach-v1",]),
            seed=tune.grid_search([1,]),
            num_resampled_goals=tune.grid_search([
                1,
            ]),
            feature_dim=tune.grid_search([
                128,
            ]),
            lr=tune.grid_search([
                6e-4,
            ]),
            critic_tau=tune.grid_search([
                0.005,
            ]),
            batch_size=tune.grid_search([
                128,
            ]),
            hidden_sizes=tune.grid_search([
                [512, 512, 512],
            ]),
            from_images=tune.grid_search([False,]),
        ),
        train_config=dict(
            num_timesteps=tune.grid_search([20_000,]),
            num_seed_steps=tune.grid_search([1_000,]),
            update_after=tune.grid_search([1_000,]),
            eval_every=tune.grid_search([1_000,]),
            gradient_steps=tune.grid_search([1,]),
        ),
    )

    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = datetime.strftime(datetime.today(), "%y-%m-%d-%H_%M")

    tune.run(
        trainable,
        name=name,
        config=config,
        num_samples=1,
        resources_per_trial={
            'cpu':1, 
#             'gpu':0.5,
        },
        local_dir=f'logs',
        log_to_file=True,
)