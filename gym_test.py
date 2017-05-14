# /usr/bin/env python3

import gym
env = gym.make('CartPole-v0')
#env = gym.make('Acrobot-v1')
#env = gym.make('MountainCar-v0')
#env = gym.make('Pendulum-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print(observation)
            print(reward, done, info)
            print("Episode finished after {} timesteps".format(t+1))
            break
