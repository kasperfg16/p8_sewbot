import gymnasium as gym
import mujoco as mj
import gym_training
import os

env = gym.make('UR5')
observation= env.reset()
#env.render(mode="rgb_array")

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   #observation, reward, terminated, truncated, info = env.step(action)
   env.step(action)
   # # Restart environment if successed or failed
   # if terminated:
   #    observation = env.reset() 
env.close()