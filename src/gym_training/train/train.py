import gymnasium as gym
import mujoco as mj
import gym_training
import os
filename = "cloth.xml"
search_path = "./"

# for root, dir, files in os.walk(search_path):
#       if filename in files:
#             model_path=os.path.abspath(os.path.join(root, filename))


# model = mj.MjModel.from_xml_path(model_path)

env = gym.make('UR3e')
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