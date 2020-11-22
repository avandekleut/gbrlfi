import gym
import numpy as np
import torch
import copy
import cv2

def KinovaWrapper(env, seed, from_images=False, fix_goals=False,):
    env = gym.wrappers.TimeLimit(env, 50)
    env = DeterministicWrapper(env, seed)
    if fix_goals:
        env = FixedGoalEnv(env)
    env = KinovaImageEnv(env) # record images
    env = DoneOnSuccessWrapper(env) 
    if from_images:
        env = LatentDistanceRewardEnv(env)
    return env

def MultiWrapper(env, seed, from_images=True, fix_goals=False):
    """
    Combine the below wrappers in the appropriate order.
    """
    env = DeterministicWrapper(env, seed)
    env = FixedViewerWrapper(env)
    if fix_goals:
        env = FixedGoalEnv(env)
    env = ImageEnv(env)
    env = DoneOnSuccessWrapper(env)
    if from_images:
        env = LatentDistanceRewardEnv(env)
    return env

class FixedViewerWrapper(gym.Wrapper):
    """
    Make camera initialization determinstic.
    """
    def __init__(self, env):
        super(FixedViewerWrapper, self).__init__(env)
        self.unwrapped._get_viewer('rgb_array')
        self.unwrapped._viewer_setup()

class DeterministicWrapper(gym.Wrapper):
    """
    Correctly seed env AND action_space. 
    """
    def __init__(self, env, seed):
        super(DeterministicWrapper, self).__init__(env)
        self.seed(seed)
        self.reset()
        
    def seed(self, seed):
        super(DeterministicWrapper, self).seed(seed)
        self.action_space.seed(seed)

class LatentDistanceRewardEnv(gym.Wrapper):
#     def __init__(self, env, agent):
    def __init__(self, env):
        super(LatentDistanceRewardEnv, self).__init__(env)
        self.encoder = None
        self.device = None
        
    def set_agent(self, agent):
        self.encoder = agent.critic_target.encoder
        self.device = agent.device
        
    def step(self, action):
        assert self.encoder is not None, "You must call `set_agent(self, agent)` before calling `step`."
        obs_dict, reward, done, info = self.env.step(action)
        reward = self.compute_reward(
            obs_dict['image_achieved_goal'],
            obs_dict['image_desired_goal'],
            dict(),
        )
        return obs_dict, reward, done, info
        
    def _to_latent(self, observation):
        with torch.no_grad():
            observation = torch.as_tensor(observation, device=self.device).unsqueeze(0)
            latent = self.encoder.forward_single_observation(observation)
            latent = latent.cpu().detach().numpy()
#             latent /= np.linalg.norm(latent)
        return latent
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        assert self.encoder is not None, "You must call `set_agent(self, agent)` before calling `compute_reward`."
        latent_achieved_goal = self._to_latent(achieved_goal)
        latent_desired_goal = self._to_latent(desired_goal)
        reward = -np.linalg.norm(latent_achieved_goal - latent_desired_goal)
#         assert -2. <= reward <= 0.
        return reward

class FixedGoalEnv(gym.Wrapper):
    """
    Wraps a gym.GoalEnv to fix the goal for learning. Good for debugging.
    """
    def __init__(self, env, fix_goal=True):
        super().__init__(env)
        self.goal = self.env.unwrapped.goal.copy()
    
    def reset(self):
        obs_dict = self.env.reset()
        obs_dict['desired_goal'] = self.goal
        self.unwrapped.goal = self.goal
        return obs_dict
    
    def step(self, action):
        obs_dict, rew, done, info = self.env.step(action)
        obs_dict['desired_goal'] = self.goal
        info.update(
            is_success=self.env.unwrapped._is_success(obs_dict['achieved_goal'], obs_dict['desired_goal'])
        )
        rew = self.env.compute_reward(obs_dict['achieved_goal'], obs_dict['desired_goal'], info)
        return obs_dict, rew, done, info

class KinovaImageEnv(gym.Wrapper):
    """
    Adds `image_observation`, `image_achieved_goal` and `image_desired_goal` fields
    to the observation dictionary. Use on real robot.
    Later: make a base class that both kinova and reach inherit from.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.env = env
        self.width = width
        self.height = height
        self.goal_img = None
        self.observation_space.spaces.update(
            image_observation=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_desired_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_achieved_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
        )
        self.reset()
        
    def reset(self):
        obs_dict = self.env.reset()
        initial_state = obs_dict['achieved_goal']
        goal = obs_dict['desired_goal']
        self.unwrapped._set_to(goal);
        self.goal_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        self.unwrapped._set_to(initial_state) # NOT "reset()" since we 
        
        obs_dict = self._update_obs_dict(obs_dict)
        
        return obs_dict
    
    def _update_obs_dict(self, obs_dict):
        obs_dict['image_desired_goal'] = self.goal_img
        obs_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        obs_dict['image_observation'] = obs_img
        obs_dict['image_achieved_goal'] = obs_img
        
        return obs_dict
        
    def _get_obs(self):
        obs_dict = self.env.env._get_obs()
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict
    
    def step(self, act):
        obs_dict, rew, done, info = self.env.step(act)
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict, rew, done, info
    
    def _set_to_goal(self, goal):
        """
        Goals are always xyz coordinates, either of gripper end effector or of object
        """
        self.unwrapped._set_to(goal)
    
class ImageEnv(gym.Wrapper):
    """
    Adds `image_observation`, `image_achieved_goal` and `image_desired_goal` fields
    to the observation dictionary. Used for rendering for now, and later for image-
    based tasks.
    """
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.env = env
        self.width = width
        self.height = height
        self.goal_img = None
        self.observation_space.spaces.update(
            image_observation=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_desired_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
            image_achieved_goal=gym.spaces.Box(0, 255, (3, self.width, self.height)),
        )
        self.reset()
        
    def reset(self):
        obs_dict = self.env.reset()
        initial_state = self.env.sim.get_state()
        goal = obs_dict['desired_goal']
        self._set_to_goal(goal);
        self.goal_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        self.env.sim.set_state(initial_state)
        
        obs_dict = self._update_obs_dict(obs_dict)
        
        return obs_dict
    
    def _update_obs_dict(self, obs_dict):
        obs_dict['image_desired_goal'] = self.goal_img
        obs_img = self.env.render(mode='rgb_array', width=self.width, height=self.height).T.copy()
        obs_dict['image_observation'] = obs_img
        obs_dict['image_achieved_goal'] = obs_img
        
        return obs_dict
        
    def _get_obs(self):
        obs_dict = self.env.env._get_obs()
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict
    
    def step(self, act):
        obs_dict, rew, done, info = self.env.step(act)
        obs_dict = self._update_obs_dict(obs_dict)
        return obs_dict, rew, done, info
    
    def _set_to_goal(self, goal):
        """
        Goals are always xyz coordinates, either of gripper end effector or of object
        """
        if self.env.has_object:
            object_qpos = self.env.sim.data.get_joint_qpos('object0:joint')
            object_qpos[:3] = goal
            self.env.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.env.sim.data.set_mocap_pos('robot0:mocap', goal)
        self.env.sim.forward()
        for _ in range(100):
            self.env.sim.step()

    

class DoneOnSuccessWrapper(gym.Wrapper):
    def __init__(self, env):
        super(DoneOnSuccessWrapper, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        return obs, reward, done, info