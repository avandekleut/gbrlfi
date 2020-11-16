# import kinova
# import gym

# kinova.register_env()
# env = gym.make("KinovaReach-v0")
# env.reset()
# for t in range(100):
#     env.step(env.action_space.sample())


import cv2
from PIL import Image

rec = cv2.VideoCapture(0)
ret, frame = rec.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
Image.fromarray(frame).save('test.png')