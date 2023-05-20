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

class UR5Env_ddpg_no_noise(MujocoEnv, EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self, **kwargs):
        
        filename = "ur5_no_noise.xml"
        search_path = "./"
        self.model_path = find_file(filename, search_path)
        self.img_width = 64
        self.img_height = 64
        self.observation_space = spaces.Box(0, 255, shape=(self.img_width, self.img_height, 3), dtype=np.uint8)

        MujocoEnv.__init__(
            self, self.model_path, 5, observation_space=self.observation_space, **kwargs
        )
        EzPickle.__init__(self, **kwargs)

        # Action space (In this case - joint limits)
        # note: The action space has dtype=uint16. The first 6 values are divided with 1000 in the step() to create a set of joint angles with precision of 0.001 between high and low values
        # self.act_space_low = np.array([
        #     -np.deg2rad(160)*1000,
        #     -np.deg2rad(45)*1000,
        #     -np.deg2rad(-90)*1000,
        #     -np.pi*1000,
        #     -np.pi*1000,
        #     -np.pi*1000,
        #     0,
        #     0]).astype(np.int16)
        
        # self.act_space_high = np.array([
        #     np.deg2rad(-110)*1000,
        #     np.deg2rad(0)*1000,
        #     np.deg2rad(90+70)*1000,
        #     np.pi*1000,
        #     np.pi*1000,
        #     np.pi*1000,
        #     2,
        #     1]).astype(np.int16)
            

        # self.action_space = spaces.Box(low=self.act_space_low, high=self.act_space_high, shape=(8,), seed=42, dtype=np.int16)
        
        self.act_space_low = np.array([
            -np.deg2rad(160),
            -np.deg2rad(45),
            -np.deg2rad(-90),
            -np.pi,
            -np.pi,
            -np.pi,
            0,
            0]).astype(np.float16)
        
        self.act_space_high = np.array([
            np.deg2rad(-110),
            np.deg2rad(0),
            np.deg2rad(90+70),
            np.pi,
            np.pi,
            np.pi,
            1,
            1]).astype(np.float16)
        
        self.action_space = spaces.Box(low=self.act_space_low, high=self.act_space_high, shape=(8,), seed=42, dtype=np.float16)
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

        self.max_step = 20

        if not self.quiet:
            self.controller.show_model_info()

    def step(self, action):
        
        # init
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.info = {}
        self.gripper_state = self.get_gripper_state(action[6])
        self.done_signal = self.to_bool(action[7])
        #action = np.float32(action/1000)
        self.joint_position = action[:6]

        if not self.quiet:
            print('done_signal: ', self.done_signal)
        
        # Check if cloth is already in good position
        if self.step_counter == 0:
            # Stay and let cloth fall
            self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(2000, render=not self.headless_mode)
            self.check()

        # Perform 'self.max_steps' movements based on action guesses from agent and then move to home pose
        self.unfold()

        # Compute reward after operation
        self.compute_reward()

        self.step_counter += 1

        if self.max_step <= self.step_counter:
            self.truncated = True

        # Recive observation after action - Before taking next action
        self.observation  = self._get_obs()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def _get_obs(self):
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")

        # # Resize img to be in observation space (For memory-efficiency)
        new_size = (self.img_width, self.img_width)
        obs = cv2.resize(image, new_size)

        # show image if headless mode is not activated
        if not self.headless_mode:
            cv2.imshow('Img used in observation space', obs)
            cv2.waitKey(1)

        # Convert to a more memory-efficient format
        obs = obs.astype(np.uint8)

        return obs

    def reset_model(self):

        self.move_reward = 0
        self.step_counter = 0
        observation = self._get_obs()

        # Initialize manipulator in home pose
        self.data.qpos[self.controller.actuated_joint_ids] = self.home_pose

        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def to_bool(self, number):
        threshold = 0.5
        if 0 <= number <= 1:
            return number > threshold
        else:
            print('Number should be a uint between 0 and 1')
    
    def get_gripper_state(self, number):
        if 0 <= number <= 2:

            if number == 0:
                return 0
            elif number == 1:
                return 1
            else:
                return 2
        else:
            print('Number should be a uint between 0 and 2')
    
    def check(self):
        self.get_coverage()
        if self.goalcoverage:
            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode, tolerance=0.02)
            self.terminated = True
            self.goalcoverage = False
    
    def unfold(self):
        
        #self.done_signal = True

        if not self.done_signal:
            result_move = self.controller.move_group_to_joint_target(target=self.joint_position, quiet=self.quiet, render=not self.headless_mode, group='Arm')
            self.move_reward += result_move

            self.controller.stay(200, render=not self.headless_mode)
            if self.gripper_state == 0:
                self.controller.close_gripper(render=not self.headless_mode)
            elif self.gripper_state == 1:
                self.controller.open_gripper(render=not self.headless_mode)
            
            self.controller.stay(50, render=not self.headless_mode)
        else:
            
            #self.show_action_space()

            # Open gripper and let textile fall
            if self.step_counter > 0:
                self.controller.open_gripper(render=not self.headless_mode)
                self.controller.stay(1000, render=not self.headless_mode)

            result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            self.move_reward += result_move

            self.controller.stay(10, render=not self.headless_mode)
    
    def compute_reward(self):# Define all the rewards possible           
        coveragereward = 0
        done_reward = 0

        # Reward weights
        w1 = 1
        w2 = 100

        if self.done_signal:
            coveragereward = self.get_coverage() # output percentage
            
            # If it says it is done but it isn't it fails (truncate)
            # If agent says it is done and it is the it has success (terminate)
            if not self.goalcoverage:
                self.truncated = True
                done_reward = -10
                
            else:
                self.terminated = True
                done_reward = 1000
        
        # Incentivice to do it faster
        step_penalty = self.step_counter/2

        # Summarize all rewards
        self.reward = self.move_reward*w1 + coveragereward*w2 + done_reward - step_penalty

        if not self.quiet:
            print('done_reward: ', done_reward)
            print('coveragereward with weight: ', coveragereward*w2)
            print('move_reward with weight: ', self.move_reward*w1)
            print('step_penalty:', -step_penalty)
            print('self.reward:', self.reward)
        
        self.move_reward = 0
    
    def get_coverage(self):
        currentarea = 0
        max_clotharea = 42483.0
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")
        self.image = image.astype(dtype=np.uint8)
        
        ## use area from ground truth      
        if self.model.skin_matid == 7: # denim
            self.image = cv2.GaussianBlur(self.image, (7, 7), 3)

        edged = cv2.Canny(self.image, 10, 250)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new2 = cv2.drawContours(self.image, contours, -1, (0,255,0), 3)

        # Show image if headless mode is not activated
        if not self.headless_mode:
            cv2.imshow('Vision system', new2)
            cv2.waitKey(1)

        if len(contours)>0:
            currentarea = cv2.contourArea(contours[0])

            ## Determine if contour is rectangular
            peri = cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], 0.04 * peri, True)
            # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle

            shape = "square" if ar >= 0.98 and ar <= 1.02 else "rectangle"

            if shape == "square" and currentarea>max_clotharea*0.90:
                if not self.quiet:
                    print("Yay, the cloth is unfolded")
                self.goalcoverage = True

        coveragereward = currentarea/max_clotharea

        return coveragereward
    
    def show_action_space(self):

        np.array([-np.pi/2, 0, np.pi/2, 0, np.pi/2, 0, 0, 0])

        target = [self.act_space_low[0]/1000, self.act_space_high[1]/1000, np.pi/2, 0, 0, self.act_space_low[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], np.pi/2, 0, 0, self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], np.pi/2, 0, self.act_space_low[4], 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], np.pi/2, 0, self.act_space_high[4], 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], np.pi/2, self.act_space_high[3], 0, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], np.pi/2, self.act_space_low[3], 0, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)
        
        target = [self.act_space_low[0], self.act_space_high[1], self.act_space_low[2], self.act_space_high[3], self.act_space_low[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], self.act_space_low[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], self.act_space_low[2], self.act_space_low[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], self.act_space_low[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_high[0], self.act_space_high[1], self.act_space_low[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_high[0], self.act_space_low[1], self.act_space_low[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_low[1], self.act_space_low[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0], self.act_space_high[1], self.act_space_high[2], self.act_space_high[3], self.act_space_high[4], self.act_space_high[5]]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)