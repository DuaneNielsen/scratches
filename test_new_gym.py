import gym
import gym_dreamer
env = gym.make('dreamer-v0')

env.reset()

observation, action, reward, info = env.step(0)