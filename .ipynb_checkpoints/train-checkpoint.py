import os
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

import kinova
kinova.register_env()

class Experiment(object):
    def __init__(
            self,
            # environment
            env_id,
        
            # visual?
#             from_images=True,
            from_images=False,

            # reproducibility
            seed=1,

            # env
            fix_goals=False,

            # compute
#             device='cuda',
            device='cpu',

            # replay buffer
            num_resampled_goals=1,
            capacity=1_000_000,

            # agent
            feature_dim=32,
            hidden_sizes=[512, 512, 512],
            log_std_bounds=[-10, 2],
            discount=0.95,
            init_temperature=0.1,
            lr=0.0006,
            actor_update_frequency=2,
            critic_tau=0.005,
            critic_target_update_frequency=2,
            batch_size=128,
        ):
        self.observation_key = 'image_observation' if from_images else 'observation'
        self.achieved_goal_key = 'image_achieved_goal' if from_images else 'achieved_goal'
        self.desired_goal_key = 'image_desired_goal' if from_images else 'desired_goal'
        
        # Seed
        utils.set_seed_everywhere(seed)

#         # Create env
        self.env = gym.make(env_id)
        if 'Kinova' in env_id:
            self.env = wrappers.KinovaWrapper(self.env, seed, from_images, fix_goals)
        else:
            self.env = wrappers.MultiWrapper(self.env, seed, from_images, fix_goals)

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
        if from_images:
#             self.env = wrappers.LatentDistanceRewardEnv(self.env, self.agent)
            self.env.set_agent(self.agent) # set the conv encoder for latent distance rewards
#         self.env.seed(seed)

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
    
    def eval(self,
            num_eval_episodes=5,
        ):
        
        average_episode_reward = 0
        average_episode_success = 0
        
        video_recorder = VideoRecorder()
        video_recorder.init()
        
        for episode in range(num_eval_episodes):
            
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
            
            average_episode_reward += episode_reward/num_eval_episodes
            average_episode_success += float(info['is_success'])/num_eval_episodes
            
        video_recorder.save(f'{self.step}.mp4')
        
        tune.report(
            eval_reward=average_episode_reward,
            eval_is_success=average_episode_success,
            timesteps_this_iter=0,
        )
        

    
    def train(self,
            gradient_steps=1, # better for wall clock time. increase for better performance.
            num_timesteps=4_000_000, # maximum time steps
            num_seed_steps=10_000, # random actions to improve exploration
            update_after=1_000, # when to start updating (off-policy still learns from seed steps)
            eval_every=10, # episodic frequency for evaluation
        ):
        
        episode = 0
        
        while self.step < num_timesteps:
            
            obs_dict = self.env.reset()
            obs = obs_dict[self.observation_key]
            obs_g = obs_dict[self.desired_goal_key]
            done = False
            episode_reward = 0
            episode_step = 0

            while not done:
                if self.step < num_seed_steps:
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
                
                if self.step >= update_after:
                    for gradient_step in range(gradient_steps):
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

            if episode % eval_every == 0:
                self.eval()
            episode += 1

                

        
        # one final test
        self.eval()