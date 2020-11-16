

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# import kortex_api
# print(kortex_api.__file__)

import utilities

import gym
import numpy as np
from gym.envs.registration import register

def register_env():
    register(
        id='KinovaReach-v0',
        entry_point='kinova:KinovaRobotEnv',
    )

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20


# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
#         print("EVENT : " + \
#               Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

class KinovaRobotEnv:
    metadata = {'render_modes':['rbg_array']}
    
    def __init__(self, action_scale=0.05, target_range=0.5, distance_threshold=0.05, speed=0.05):
        # Parse arguments to create connections later
        self.connection = utilities.DeviceConnection.createTcpConnection()
        self.router = self.connection.__enter__()
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        
        # Env params
        # action_scale multiplies actions to make them smaller
        # target_range gives the +/- distance that goals are sampled from initial gripper pos. 
        # distance_threshold is used for success criterion and sparse reward.
        self.action_scale = action_scale
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.speed = speed
        
        self.observation_space = gym.spaces.Dict(
            observation=gym.spaces.Box(-np.inf, np.inf, (12,)),
            achieved_goal=gym.spaces.Box(-np.inf, np.inf, (3,)),
            desired_goal=gym.spaces.Box(-np.inf, np.inf, (3,)),
            
        )
        self.action_space = gym.spaces.Box(-1., 1., (3,))
        
        self.reward_range = (-np.inf, 0)
        self.unwrapped = self
        self.spec = None
        
        self._set_to_home()
        self.init_xyz = self._get_obs()[:3] # first three entries are xyz pos of EE.
        self.goal = None
        self.sample_goal()
        assert self.goal is not None
        
    def seed(self, seed):
        pass
        
    def close(self):        
        self.connection.__exit__(None, None, None)
        
    def render(self, mode='rgb_array'):
        return np.zeros(3, 84, 84)
        
    def sample_goal(self, target_range=None):
        if target_range is None:
            target_range = self.target_range
        self.goal = self.init_xyz + (-1 + 2*np.random.rand(*self.action_space.shape))*self.target_range
        return self.goal
    
    def reset(self):
        self.sample_goal()
        self._set_to_home()
        return self._get_obs_dict()
    
    def step(self, act):
        act *= self.action_scale
            
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

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteAction(action)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
            
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
    
    def _set_to_home(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Move arm to ready position
#             print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            raise RuntimeError("Could not reach a safe position.")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            
    def _get_obs(self):

        feedback = self.base_cyclic.RefreshFeedback()

        observation = np.asarray([
            feedback.base.tool_pose_x,
            feedback.base.tool_pose_y,
            feedback.base.tool_pose_z,
            feedback.base.tool_pose_theta_x,
            feedback.base.tool_pose_theta_y,
            feedback.base.tool_pose_theta_z,
            feedback.base.tool_twist_linear_x,
            feedback.base.tool_twist_linear_y,
            feedback.base.tool_twist_linear_z,
            feedback.base.tool_twist_angular_x,
            feedback.base.tool_twist_angular_y,
            feedback.base.tool_twist_angular_z,
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