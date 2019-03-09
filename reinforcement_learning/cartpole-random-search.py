import gym
import numpy as np

env = gym.make('CartPole-v1')
parameters = np.random.rand(4) * 2 - 1


def run_episode(env, parameters):
    """
    a determinstic random policy using a simple matrix multiplication of the input
    :return: total reward
    """
    total_reward = 0
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = 0 if np.matmul(parameters,observation) < 0 else 1
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    return total_reward

bestparams = np.load('cartpole_params.npy')
bestreward = 0

reward = run_episode(env, bestparams)

# for _ in range(10000):
#     parameters = np.random.rand(4) * 2 - 1
#     reward = run_episode(env,parameters)
#     if reward > bestreward:
#         bestreward = reward
#         bestparams = parameters
#         np.save('cartpole_params', bestparams)
#         # considered solved if the agent lasts 200 timesteps
#         if reward == 200:
#             break

