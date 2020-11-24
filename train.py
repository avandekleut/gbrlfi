import os
import glob
import random
import copy

from video import VideoRecorder
from agent import Agent as Agent
from buffer import HindsightReplayBuffer
import utils
import wrappers

import time
import sys
from datetime import datetime

import gym

import torch

from ray import tune

try:
    import kinova
    kinova.register_env()
except:
    print('Kinova not found.')

import pickle

class Experiment(object):
    def __init__(
            self,
            # environment
            env_id,
        
            # visual?
            from_images=False,

            # reproducibility
            seed=1,

            # env
            fix_goals=False,

            # compute
            device='cuda' if torch.cuda.is_available() else 'cpu',

            # replay buffer
            num_resampled_goals=1,
            capacity=1_000_000,

            # agent
            feature_dim=128,
            hidden_sizes=[512, 512, 512],
            log_std_bounds=[-10, 2],
            discount=0.95,
            init_temperature=0.1,
            lr=0.0006,
            actor_update_frequency=2,
            critic_tau=0.005,
            critic_target_update_frequency=2,
            batch_size=128,
        
            # evaluation
            num_eval_episodes=5,
            
            # training
            gradient_steps=1, # better for wall clock time. increase for better performance.
            num_timesteps=20_000, # maximum time steps
            num_seed_steps=1_000, # random actions to improve exploration
            update_after=1_000, # when to start updating (off-policy still learns from seed steps)
            eval_every=20, # episodic frequency for evaluation
            save_every=5_000, # how often to save the experiment progress in time steps
        
            **kwargs, # lazily absorb extra args
        
        ):
        self.observation_key = 'image_observation' if from_images else 'observation'
        self.achieved_goal_key = 'image_achieved_goal' if from_images else 'achieved_goal'
        self.desired_goal_key = 'image_desired_goal' if from_images else 'desired_goal'
        
        # Seed
        utils.set_seed_everywhere(seed)

#         # Create env
        self.env_id = env_id
        self.seed = seed
        self.from_images = from_images
        self.fix_goals = fix_goals
        
        self.env = gym.make(self.env_id)
        if 'Kinova' in self.env_id:
            self.env = wrappers.KinovaWrapper(self.env, self.seed, self.from_images, self.fix_goals)
        else:
            self.env = wrappers.MultiWrapper(self.env, self.seed, self.from_images, self.fix_goals)

        # Create agent
        self.agent = Agent(
            from_images,
            self.env.observation_space,
            self.env.action_space,
            device=device, 
            feature_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            log_std_bounds=log_std_bounds,
            discount=discount,
            init_temperature=init_temperature,
            lr=lr,
            critic_tau=critic_tau,
            batch_size=batch_size
        )
        
        # update env to use agent encoder for images if necessary
        if self.from_images:
            self.env.set_agent(self.agent) # set the conv encoder for latent distance rewards

        # Create replay buffer
        self.replay_buffer = HindsightReplayBuffer(
            from_images=from_images,
            env=self.env,
            num_resampled_goals=num_resampled_goals,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            capacity=capacity,
            device=device, 
        )

        self.step = 0
        self.num_eval_episodes = num_eval_episodes
        
        self.gradient_steps = gradient_steps
        self.num_timesteps = num_timesteps
        self.num_seed_steps = num_seed_steps
        self.update_after = update_after
        self.eval_every = eval_every
        self.save_every = save_every
    
    def eval(self):
        average_episode_reward = 0
        average_episode_success = 0
        
        video_recorder = VideoRecorder()
        video_recorder.init()
        
        for episode in range(self.num_eval_episodes):
            
            obs_dict = self.env.reset()
            obs = obs_dict[self.observation_key]
            obs_g = obs_dict[self.desired_goal_key]
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                action = self.agent.act(obs, obs_g, sample=True)

                next_obs_dict, reward, done, info = self.env.step(action)

                done = float(done)
                episode_reward += reward

                achieved_goal = next_obs_dict[self.achieved_goal_key]

                obs = next_obs_dict[self.observation_key]
                obs_g = next_obs_dict[self.desired_goal_key]
                episode_step += 1
                
                video_recorder.record(next_obs_dict)
            
            average_episode_reward += episode_reward/self.num_eval_episodes
            average_episode_success += float(info['is_success'])/self.num_eval_episodes
            
        video_recorder.save(f'{self.step}.mp4')
        
        tune.report(
            eval_reward=average_episode_reward,
            eval_is_success=average_episode_success,
            timesteps_this_iter=0,
        )
        
    
    def train(self):
        episode = 0
        
        while self.step < self.num_timesteps:
            
            obs_dict = self.env.reset()
            obs = obs_dict[self.observation_key]
            obs_g = obs_dict[self.desired_goal_key]
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                if self.step % self.save_every == 0:
                    self.save(f'checkpoints/{self.step}.tar')
#                     self.load(f'checkpoints/{self.step}.tar')
                    
                if self.step < self.num_seed_steps:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.act(obs, obs_g, sample=True)

                next_obs_dict, reward, done, info = self.env.step(action)
                next_obs = next_obs_dict[self.observation_key]

                # Allow infinite bootstrap:
                # If the episode was cut off due to time limit, consider done to be false
                done = float(done)
                done_no_max = 0 if episode_step + 1 == self.env.spec.max_episode_steps else done
                episode_reward += reward

                achieved_goal = next_obs_dict[self.achieved_goal_key]

                self.replay_buffer.add(obs, obs_g, achieved_goal, action, reward, next_obs, done, done_no_max)
                
                if self.step >= self.update_after:
                    for gradient_step in range(self.gradient_steps):
                        self.agent.update(self.replay_buffer, gradient_step)

                obs = next_obs_dict[self.observation_key]
                obs_g = next_obs_dict[self.desired_goal_key]
                episode_step += 1
                self.step += 1
            
            tune.report(
                train_reward=episode_reward,
                train_is_success=float(info['is_success']),
                timesteps_this_iter=episode_step,
                **self.agent.info,
            )

            if episode % self.eval_every == 0:
                self.eval()
            episode += 1

        # one final test
        self.eval()
        
    def save(self, path):
        dirs = os.path.dirname(path)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
#         with open(path, 'wb+') as f:
#             pickle.dump(self, f)
        
    def load(self, path):
        print(f'Resuming from {path}')
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.__dict__.update(saved.__dict__)
            
            # envs don't save correctly via pickle so re-make it.
            self.env = gym.make(self.env_id)
            if 'Kinova' in self.env_id:
                self.env = wrappers.KinovaWrapper(self.env, self.seed, self.from_images, self.fix_goals)
            else:
                self.env = wrappers.MultiWrapper(self.env, self.seed, self.from_images, self.fix_goals)
        
            # update env to use agent encoder for images if necessary
            if self.from_images:
                self.env.set_agent(self.agent) # set the conv encoder for latent distance rewards

            
#     def load_most_recent(self):
#         """
#         Locate and load most recent checkpoint
#         """
#         list_of_files = glob.glob('checkpoints/*')
#         if len(list_of_files) > 0:
#             latest_file = max(list_of_files, key=os.path.getctime)
#             self.load(latest_file)
#         else:
#             print('No recent checkpoints.')