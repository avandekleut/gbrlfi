import sys
import os
import time
import threading
import time

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

import utilities

import gym
import numpy as np
from gym.envs.registration import register

import cv2

def register_env():
    register(
        id='KinovaReach-v0',
        entry_point='kinova:KinovaRobotEnv',
    )

class KinovaRobotEnv:
    metadata = {'render_modes':['human', 'rbg_array']}
    
    def __init__(self, action_scale=0.05, target_range=0.15, distance_threshold=0.05, speed=0.10):
        self.connection = utilities.DeviceConnection.createTcpConnection()
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        

        self.webcam = cv2.VideoCapture(0)
        
        self.action_scale = action_scale
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.speed = speed
        
        self.observation_space = gym.spaces.Dict(
            observation=gym.spaces.Box(-np.inf, np.inf, self._get_obs().shape),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, (3,)),
            desired_goal=gym.spaces.Box(-np.inf, np.inf, (3,)),
            
        )
        self.action_space = gym.spaces.Box(-1., 1., (3,))
        
        self.reward_range = (-np.inf, 0)
        self.unwrapped = self
        self.spec = None
        
        self._set_to_home()
        self.init_xyz = self._get_obs()[:3]
        self.goal = None
        self.sample_goal()
        assert self.goal is not None
        
    def seed(self, seed):
        pass
        
    def close(self):        
        self.connection.__exit__(None, None, None)
        
    def render(self, mode='human', width=84, height=84):
        ret, frame = self.webcam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mode == 'human':
            cv2.imshow('', frame)
        elif mode == 'rgb_array':
            frame = cv2.resize(frame, (width, height))
            return frame
        
    def sample_goal(self, target_range=None):
        if target_range is None:
            target_range = self.target_range
        self.goal = self.init_xyz + (-1 + 2*np.random.rand(*self.init_xyz.shape))*self.target_range
        return self.goal
    
    def reset(self):
        self.sample_goal()
        self._set_to_home()
        return self._get_obs_dict()
    
    def step(self, act):
        act = act.copy()*self.action_scale
            
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        action.reach_pose.constraint.speed.translation = self.speed

        feedback = self.base_cyclic.RefreshFeedback()

        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x + act[0]          # (meters)
        cartesian_pose.y = feedback.base.tool_pose_y + act[1]    # (meters)
        cartesian_pose.z = feedback.base.tool_pose_z + act[2]    # (meters)
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x  # (degrees)
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

#         e = threading.Event()
#         notification_handle = self.base.OnNotificationActionTopic(
#             check_for_end_or_abort(e),
#             Base_pb2.NotificationOptions()
#         )

        self.base.ExecuteAction(action)
#         finished = e.wait(TIMEOUT_DURATION)
#         self.base.Unsubscribe(notification_handle)

        duration = (np.linalg.norm(act)/self.speed)*2
        print(f'waiting w duration {duration}')
        time.sleep(duration)
            
        obs_dict = self._get_obs_dict()
        info = dict()
        reward = self.compute_reward(
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
            info,
        )
        done = True if np.abs(reward) < self.distance_threshold else False
        info['is_success'] = done
        return obs_dict, reward, done, info
        
    def _set_to(self, xyz):
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""
        
        action.reach_pose.constraint.speed.translation = self.speed

        feedback = self.base_cyclic.RefreshFeedback()
        
        current_pos = np.asarray([feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z])
        
        distance_to_init_xyz = np.linalg.norm(current_pos - self.init_xyz)
        duration = 2.0*distance_to_init_xyz/self.speed
        print(f'resetting w duration {duration}')
        
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = xyz[0]
        cartesian_pose.y = xyz[1]
        cartesian_pose.z = xyz[2]
#         cartesian_pose.theta_x = -180
#         cartesian_pose.theta_y = 0
#         cartesian_pose.theta_z = 90

#         e = threading.Event()
#         notification_handle = self.base.OnNotificationActionTopic(
#             check_for_end_or_abort(e),
#             Base_pb2.NotificationOptions()
#         )

        self.base.ExecuteAction(action)
        time.sleep(duration)
#         finished = e.wait(TIMEOUT_DURATION)
#         self.base.Unsubscribe(notification_handle)

    def _set_to_home(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            raise Run("Can't reach safe position. Exiting")

        
        self.base.ExecuteActionFromReference(action_handle)
        print('Resetting for duration 6')
        time.sleep(6)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return 0. if d < self.distance_threshold else -1.
    
    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return True if d < self.distance_threshold else False
            
    def _get_obs(self):

        feedback = self.base_cyclic.RefreshFeedback()

        observation = np.asarray([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
#             feedback.base.tool_pose_theta_x,
#             feedback.base.tool_pose_theta_y,
#             feedback.base.tool_pose_theta_z,
#             feedback.base.tool_twist_linear_x,
#             feedback.base.tool_twist_linear_y,
#             feedback.base.tool_twist_linear_z,
#             feedback.base.tool_twist_angular_x,
#             feedback.base.tool_twist_angular_y,
#             feedback.base.tool_twist_angular_z,
        ])

        return observation
        
    def _get_obs_dict(self):
        observation = self._get_obs()
        achieved_goal = observation[:3]
        desired_goal = self.goal

        return dict(
            observation=observation,
            achieved_goal=achieved_goal,
            desired_goal=desired_goal,
        )
        
            
    
if __name__ == "__main__":
    env = KinovaRobotEnv()
    import wrappers
#     env = wrappers.KinovaWrapper(env, 1, from_images=True, fix_goals=False)