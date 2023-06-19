import math
from typing import Optional
import numpy as np
import gymnasium as gym
import random
import gymnasium as gym
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gym_training.controller.controller_simple import MJ_Controller
import mujoco.viewer
import time

def find_file(filename, search_path):
    """
    Search for a file in the given search path.
    Returns the full path to the file if found, or None otherwise.
    """
    
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

class UR5Env_ddpg_push(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):

        filename = "ur5_simple_touch.xml"
        search_path = "./"
        self.model_path = find_file(filename, search_path)
        self.img_width = 64
        self.img_height = 64

        # Action space (In this case - joint limits)
        # note: The action space has dtype=uint16. The first 6 values are divided with 1000 in the step() to create a set of joint angles with precision of 0.001 between high and low values
        self.descretization = 100
        self.act_space_low = np.array([
            -np.deg2rad(90)*self.descretization,# -np.deg2rad(150)*self.descretization,
            np.deg2rad(0)*self.descretization,# -np.deg2rad(45)*self.descretization,
            np.deg2rad(-90)*self.descretization,# -np.deg2rad(-90)*self.descretization,
            np.deg2rad(-90)*self.descretization,# -np.deg2rad(90)*self.descretization
            ]).astype(np.int16)
        
        self.act_space_high = np.array([
            -np.deg2rad(-90)*self.descretization,#np.deg2rad(-110)*self.descretization,
            np.deg2rad(45)*self.descretization,#np.deg2rad(-30)*self.descretization,
            np.deg2rad(45)*self.descretization,#np.deg2rad(90+60)*self.descretization,
            np.deg2rad(30)*self.descretization#-np.deg2rad(0)*self.descretization
            ]).astype(np.int16)
                
        self.observation_space = spaces.Box(min(self.act_space_low), max(self.act_space_high), shape=(7,), dtype=np.float32)
            
        self.action_space = spaces.Box(low=self.act_space_low, high=self.act_space_high, shape=(4,), seed=42, dtype=np.int16)

        MujocoEnv.__init__(
            self, self.model_path, 2, observation_space=self.observation_space, **kwargs
        )

        EzPickle.__init__(self, **kwargs)
        
        self.controller = MJ_Controller(model=self.model, data=self.data, mujoco_renderer=self.mujoco_renderer)
        self.step_counter = 0
        self.distance_goal = 0
        self.goalcoverage = False
        self.area_stack = [0]*2
        self.move_reward = 0
        self.home_pose = np.array([0, 0, 0, 0])
        self.image = np.empty((480, 480, 3)) # image size
        self.in_home_pose = False

        # Do not show renders?
        self.headless_mode = True

        # Do not print output in terminal?
        self.quiet = True

        self.max_step = self.frame_skip*3

    def step(self, action):

        self.info = {}
        self.do_simulation(action, self.frame_skip)
        time.sleep(0.1)
        if not self.headless_mode:
            self.mujoco_renderer.render("human")
        self.step_counter += 1

        # ### Forstå hvorfor dette er nødvendigt
        if self.max_step == self.step_counter:
            print("max steps")
            self.truncated = True
        
        # Compute reward after operation
        self.compute_reward()

        # Recive observation after action - Before taking next action
        self.observation  = self._get_obs()
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def _get_obs(self):
        # Joint positions
        joint_pos = self.data.qpos[self.controller.actuated_joint_ids].copy()

        ## Textile pos
        textile_pos = self.model.geom_pos[1] #[-0.2, 0.4, 0.51]
        

        # Concatenate joint_pos and image arrays along axis 0
        concatenated_array = np.concatenate([joint_pos, textile_pos]).astype(np.float32)

        return concatenated_array

    def reset_model(self):
        print("\n################################################")
        print("RESETTING")
        print("################################################\n")

        self.move_reward = 0
        self.step_counter = 0
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.done_signal = False

        # Initialize manipulator in home pose
        self.data.qpos[self.controller.actuated_joint_ids] = self.home_pose
        observation = self._get_obs()

        return observation

    def compute_reward(self):# Define all the rewards possible      
        touchreward = 0

        #textile_center = [-0.2, 0.4, 0.51]
        goal_distance = [-0.4, 0.4, 0.51]
        self.distance_goal = abs(math.dist(np.ndarray.tolist(self.model.geom_pos[1]), goal_distance))

        distance_edge = abs(math.dist(np.ndarray.tolist(self.data.xpos[12]), (np.subtract(np.ndarray.tolist(self.model.geom_pos[1]), [0.1, 0, 0]))))

        if self.distance_goal == 0:
            print("\n################################################")
            print("MOVED TEXTILE")
            print("##################################################\n")
            self.terminated = True

        elif 1 in np.ndarray.tolist(self.data.contact.geom1) and all(x in np.ndarray.tolist(self.data.contact.geom2) for x in [28, 30]):
            touchreward = 100

        elif 1 in np.ndarray.tolist(self.data.contact.geom1) and 28 in np.ndarray.tolist(self.data.contact.geom2):
            touchreward = 100
            
        elif 1 in np.ndarray.tolist(self.data.contact.geom1) and 30 in np.ndarray.tolist(self.data.contact.geom2):
            touchreward = 100
            
        elif len(np.ndarray.tolist(self.data.contact.geom1))==0 and len(np.ndarray.tolist(self.data.contact.geom2))==0:
            None
        else:
            print("Collision!")
            self.truncated = True
        
        # Summarize all rewardser
        if self.truncated == True:
            self.reward =  -500
        elif self.terminated == True:
            self.reward = 1000
        else:
            self.reward = (20 - pow(self.distance_goal*10, 2)) + (20-pow(distance_edge*10, 2)) + touchreward #+ orientationreward

        print("Reward: ", self.reward)
    
    def _set_action_space(self):
        self.action_space = self.action_space
