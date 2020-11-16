import os
import sys

import imageio
import numpy as np
import cv2

import utils


class VideoRecorder(object):
    def __init__(self, height=256, width=256, fps=10):
        self.save_dir = os.path.join(os.getcwd(), 'video')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = enabled
        
    def make_frame(self, obs_dict):
        obs = obs_dict['image_achieved_goal']
        obs_g = obs_dict['image_desired_goal']
        
        frame = np.concatenate((obs, obs_g), axis=1)
        frame = frame.T
        frame = cv2.resize(frame, (2*self.width, self.height), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        return frame

    def record(self, obs_dict):
        if self.enabled:
            frame = self.make_frame(obs_dict)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            import time
            t0 = time.time()
            imageio.mimsave(path, self.frames, fps=self.fps)
            t1 = time.time()
