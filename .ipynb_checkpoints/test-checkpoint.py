import gym
import wrappers
env = gym.make("FetchReach-v1")
env = wrappers.MultiWrapper(env, 1, from_images=True, fix_goals=False)
agent = Agent(from_images=True, observation_space=env.observation_space, action_space=env.action_space)