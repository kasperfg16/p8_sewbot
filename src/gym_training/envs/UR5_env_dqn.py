"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

import math
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
from gym_training.controller.mujoco_controller import MJ_Controller
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

class CartPoleEnv_kasper(MujocoEnv, EzPickle):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, render_mode: Optional[str] = None, **kwargs):

        filename = "ur5_no_noise.xml"
        search_path = "./"
        self.model_path = find_file(filename, search_path)
        self.img_width = 64
        self.img_height = 64

        self.descretization = 100

        self.act_space_low = np.array([
            -np.deg2rad(150)*self.descretization
            ]).astype(np.int16)
        
        self.act_space_high = np.array([
            np.deg2rad(-110)*self.descretization
            ]).astype(np.int16)
                
        self.observation_space = spaces.Box(min(self.act_space_low), max(-self.act_space_high), shape=(self.img_width*self.img_height*3+8, ), dtype=np.float32)
            
        # Move robot clockwise or counter-clockwise
        self.action_space = spaces.Discrete(2)

        MujocoEnv.__init__(
            self, self.model_path, 5, observation_space=self.observation_space, **kwargs
        )

        EzPickle.__init__(self, **kwargs)
        
        self.controller = MJ_Controller(model=self.model, data=self.data, mujoco_renderer=self.mujoco_renderer)
        self.step_counter = 0
        self.stepcount = 0
        self.goalcoverage = False
        self.area_stack = [0]*2
        self.move_reward = 0
        self.home_pose = np.array([-np.pi/2, 0, np.pi/2, 0, np.pi/2, 0, 0, 0])
        self.image = np.empty((480, 480, 3)) # image size
        self.in_home_pose = False

        # Do not show renders?
        self.headless_mode = True

        # Do not print output in terminal?
        self.quiet = True

        self.max_step = 100

        if not self.quiet:
            #self.controller.show_model_info()
            1

    def step(self, action):
        
        # init
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.info = {}

        increase = 5

        if action == 0:
            # current joint pos -
            current_pos = self.data.qpos[self.controller.actuated_joint_ids].copy()
            joint1_pos = current_pos[0]+np.deg2rad(-increase)
        elif action == 1:
            # current joint pos +
            current_pos = self.data.qpos[self.controller.actuated_joint_ids].copy()
            joint1_pos = current_pos[0]+np.deg2rad(increase)
        else:
            print('Something went wrong!')

        self.joint_position = np.array([joint1_pos, 0, np.pi/2, 0, np.pi/2, 0])
        self.result_move = self.controller.move_group_to_joint_target(target=self.joint_position, quiet=self.quiet, render=not self.headless_mode, group="Arm")

        self.step_counter += 1

        if self.max_step <= self.step_counter:
            self.truncated = True
        
        # Compute reward after operation
        self.compute_reward()

        # Recive observation after action - Before taking next action
        self.observation  = self._get_obs()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self):
        self.step_counter = 0

        # Initialize manipulator in home pose
        self.data.qpos[self.controller.actuated_joint_ids] = self.home_pose
        
        # Get joint positions and image as observation
        observation = self._get_obs()

        return observation, {}
    
    def compute_reward(self):# Define all the rewards possible           
        # Get the coordinates of Cloth
        pos=[-0.2, 0.4, .51]

        # Get the coordinates of Cloth
        cloth_x, cloth_y, cloth_z = pos

        # Get the coordinates of "right_finger"
        finger_x = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")][0]
        finger_y = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")][1]
        finger_z = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")][2]

        # Calculate the distance between the two points
        distance = math.sqrt((finger_x - cloth_x)**2 + (finger_y - cloth_y)**2 + (finger_z - cloth_z)**2)

        print('plate: ', self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")])
        print('finger: ', self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")])
        print('distance: ', distance)

        self.reward = - distance
    
    def _get_obs(self):
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")

        # Resize image to match observation space dimensions (For memory-efficiency)
        new_size = (self.img_width, self.img_width)
        image = cv2.resize(image, new_size)

        # Show image if headless mode is not activated
        if not self.headless_mode:
            cv2.imshow('Img used in observation space', image)
            cv2.waitKey(1)

        # Convert to a more memory-efficient format
        image = image.astype(np.uint8)

        # Flatten the image
        flattened_image = image.reshape(-1)

        # Joint positions
        joint_pos = self.data.qpos[self.controller.actuated_joint_ids].copy()

        # Concatenate joint_pos and image arrays along axis 0
        concatenated_array = np.concatenate([joint_pos, flattened_image]).astype(np.float32)

        return concatenated_array
    
    def _set_action_space(self):
        self.action_space = self.action_space