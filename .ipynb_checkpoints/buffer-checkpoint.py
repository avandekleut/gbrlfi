import numpy as np
import scipy.spatial
import random
import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, from_images, observation_space, action_space, capacity, device):
        prefix = 'image_' if from_images else ''
        dtype = np.uint8 if from_images else np.float32
            
        observation_shape = observation_space[prefix+'observation'].shape
        desired_goal_shape = observation_space[prefix+'desired_goal'].shape
        achieved_goal_shape = observation_space[prefix+'achieved_goal'].shape
        action_shape = action_space.shape

        self.observations = np.empty((capacity, *observation_shape), dtype=dtype)
        self.desired_goals = np.empty((capacity, *desired_goal_shape), dtype=dtype)
        self.achieved_goals = np.empty((capacity, *achieved_goal_shape), dtype=dtype)
        self.next_observations = np.empty((capacity, *observation_shape), dtype=dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        
        self.capacity = capacity
        self.device = device

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, observation, desired_goal, achieved_goal, action, reward, next_observation, done, done_no_max):
        np.copyto(self.observations[self.idx], observation)
        np.copyto(self.desired_goals[self.idx], desired_goal)
        np.copyto(self.achieved_goals[self.idx], achieved_goal)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_observations[self.idx], next_observation)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity  #Ring buffer
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self), size=batch_size)

        observations = torch.as_tensor(self.observations[idxs], device=self.device).float()
        desired_goals = torch.as_tensor(self.desired_goals[idxs], device=self.device).float()
        next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device).float()
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device).float()

        return observations, desired_goals, actions, rewards, next_observations, not_dones_no_max

class HindsightReplayBuffer(ReplayBuffer):
    """Buffer to store environment transitions. Samples HER goals online."""
    """Actually recomputes the reward always."""
    def __init__(self, from_images, env, num_resampled_goals, observation_space, action_space, capacity, device):
        super(HindsightReplayBuffer, self).__init__(from_images, observation_space, action_space, capacity, device)
        self.env = env
        self.num_resampled_goals = num_resampled_goals
        self.her_ratio = 1/(1+self.num_resampled_goals)
    
    def sample(self, batch_size):
        # find out when episodes ended
        dones = 1 - self.not_dones[:len(self)] # check dones up to current stored max
        episode_ends = np.nonzero(dones)[0]
        episode_starts = np.concatenate((np.array([0]), episode_ends[:-1]+1)) # exclude last "end" and add 1 to go to start of next ep
        
        # filter out trajectories of length 1
        # use tuple assignment to avoid dependency issues
        valid_episodes = (episode_ends - episode_starts) >= 2
        episode_stats = episode_starts[valid_episodes]
        episode_ends = episode_ends[valid_episodes]
        
        batch_indices = np.random.randint(len(episode_ends), size=batch_size)
        
        # choose the timesteps corresponding the start and end of those sampled episodes
        batch_ends = episode_ends[batch_indices]
        batch_starts = episode_starts[batch_indices]
        
        # make space for the recomputed desired_goals and rewards
        idxs = np.empty(batch_size, dtype=np.int32) # store the sampled transition indices for fast indexing later
        desired_goals = np.empty((batch_size, *self.desired_goals.shape[1:]))
        rewards = np.empty((batch_size, *self.rewards.shape[1:]))

        # iterate through each sampled episode to add to the batch
        for i, (batch_start, batch_end) in enumerate(zip(batch_starts, batch_ends)):
            # choose the transition we will resample goals for.
            transition = np.random.randint(batch_start, batch_end-1)
            # probabilistically replace the goal with a new goal.
            if np.random.rand() < self.her_ratio:
                # original goal
                desired_goal = self.desired_goals[transition]
#                 reward = self.rewards[transition]
            else:
                # resampled goal

                # choose a transition some time between the sampled one and the end of the episode
                future_transition = np.random.randint(transition+1, batch_end)
                desired_goal = self.achieved_goals[future_transition]
            achieved_goal = self.achieved_goals[transition+1] # its the next timestep's achieved goal
            reward = self.env.compute_reward(achieved_goal, desired_goal, dict())
            idxs[i] = transition
            desired_goals[i] = desired_goal
            rewards[i] = reward

        assert desired_goals.shape == self.desired_goals[idxs].shape
        assert rewards.shape == self.rewards[idxs].shape

        observations = torch.as_tensor(self.observations[idxs], device=self.device).float()
        desired_goals = torch.as_tensor(desired_goals, device=self.device).float()
        next_observations = torch.as_tensor(self.next_observations[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device).float()
        rewards = torch.as_tensor(rewards, device=self.device).float()
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device).float()

        return observations, desired_goals, actions, rewards, next_observations, not_dones_no_max