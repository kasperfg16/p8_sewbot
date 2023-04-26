import gymnasium as gym
import gym_training

env = gym.make('UR3e')

observation, info = env.reset()

for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   
   # Restart environment if successed or failed
   if terminated or truncated:
      observation, info = env.reset() 
env.close()