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

def index_difference(arr1, arr2):
    difference = []
    for i in range(len(arr1)):
        if i < len(arr2):
            diff = abs(i - arr2[i])
            difference.append(diff)
        else:
            difference.append(i)
    return difference

def calculate_product(arr):
    product = 1
    for num in arr:
        product *= num
    return product

def find_file(filename, search_path):
    """
    Search for a file in the given search path.
    Returns the full path to the file if found, or None otherwise.
    """
    
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

class UR5Env_ddpg_no_noise_simplified(MujocoEnv, EzPickle):
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

        # Action space (In this case -d joint limits)
        # note: The action space has dtype=uint16. The first 6 values are divided with 1000 in the step() to create a set of joint angles with precision of 0.001 between high and low values
        self.descretization = 100
        self.act_space_low = np.array([
            -np.deg2rad(150)*self.descretization
            ]).astype(np.int16)
        
        self.act_space_high = np.array([
            np.deg2rad(-110)*self.descretization
            ]).astype(np.int16)
        
        # Example usage:
        result = index_difference(self.act_space_low, self.act_space_high)

        print('posible actions: ', calculate_product(result))
                
        self.observation_space = spaces.Box(min(self.act_space_low), max(self.act_space_high), shape=(self.img_width*self.img_height*3+8, ), dtype=np.float32)
            
        self.action_space = spaces.Box(low=self.act_space_low, high=self.act_space_high, shape=(len(self.act_space_low),), seed=42, dtype=np.int16)

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
        self.gripper_state = 1
        self.done_signal = False
        action = np.float32(action/self.descretization)

        # Scara-like motion
#        self.joint_position = np.array([action[0], action[1], action[2], action[2]-action[1]-np.pi/2, np.pi/2, 0])
        self.joint_position = np.array([action[0], 0, np.pi/2, 0, np.pi/2, 0])
        
        if not self.quiet:
            print('done_signal: ', self.done_signal)
        
        # Check if cloth is already in good position
        if self.step_counter == 0:
            # Stay and let cloth fall
            self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(200, render=not self.headless_mode)
#            self.check()

        # Perform 'self.max_steps' movements based on action guesses from agent and then move to home pose
        self.unfold()

        self.step_counter += 1

        if self.max_step <= self.step_counter:

            #result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            #self.move_reward += result_move
            self.truncated = True
        
        # Compute reward after operation
        self.compute_reward()

        # Recive observation after action - Before taking next action
        self.observation  = self._get_obs()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

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

    def reset_model(self):

        #self.randomizationSparse()
        self.move_reward = 0
        self.step_counter = 0

        # Initialize manipulator in home pose
        self.data.qpos[self.controller.actuated_joint_ids] = self.home_pose
        observation = self._get_obs()

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
        if 0 <= number <= 1:

            if number == 0:
                return 0
            elif number == 1:
                return 1
        else:
            print('Number should be a uint between 0 and 1')
    
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
            #self.move_reward += result_move
    
    def compute_reward(self):# Define all the rewards possible           
        coveragereward = 0
        done_reward = 0

        # Reward weights
        w1 = 0
        w2 = 100
        
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

        if self.truncated: #self.done_signal or self.truncated:
            
            # Do not give coverage reward if it has not moved camera check first will ensure to not start the agent task in a real scenario
            if self.step_counter > 0:
                coveragereward = self.get_coverage() # output percentage
            
            # If it says it is done but it isn't it fails (truncate)
            # If agent says it is done and it is the it has success (terminate)
            if not self.goalcoverage:
                #self.truncated = True
                done_reward = 0
                
            else:
                self.terminated = True
                self.truncated = False
                done_reward = 1000
        
        # Incentivice to do it faster
        step_penalty = self.step_counter/2

        # Summarize all rewards
        #elf.reward = self.move_reward*w1 + coveragereward*w2 + done_reward - step_penalty - distancetocamcenter
        #self.reward = coveragereward*w2 + done_reward - distance
        self.reward = - distance

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

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, 0, 0, self.act_space_low[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, 0, 0, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, 0, self.act_space_low[4]/self.descretization, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, 0, self.act_space_high[4]/self.descretization, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, self.act_space_high[3]/self.descretization, 0, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, np.pi/2, self.act_space_low[3]/self.descretization, 0, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(4000, render=not self.headless_mode)
        
        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_low[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_low[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_high[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_high[0]/self.descretization, self.act_space_low[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_low[1]/self.descretization, self.act_space_low[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)

        target = [self.act_space_low[0]/self.descretization, self.act_space_high[1]/self.descretization, self.act_space_high[2]/self.descretization, self.act_space_high[3]/self.descretization, self.act_space_high[4]/self.descretization, self.act_space_high[5]/self.descretization]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)
    
    def _set_action_space(self):
        self.action_space = self.action_space

    def randomizationSparse(self): # Randomization between trainings

        """
        Cloth randomization:
            Cloth color (randomizing hue, saturation, value, and colors, along with lighting and glossiness.)
            Mechanical properties of cloth
                Here we use the default value and perturbates with <=+-15%
        | Name   | Number | Mass | Friction | Stiffness | Damping |
        |--------|--------|------|----------|---------- |---------|
        | Denim  |   7    | 
        | Nylon  |   8    |      |          |           |         |
        | Poly   |   9    |      |          |           |         |
        | Cotton |   10   |      |          |           |         |
        | Cotton |   11   |      |          |           |         |

            matdenim = 7, matwhite1 = 8, matwhite2 = 9, matwhite4 = 10, matblack = 11
        """
        
        cloth_id = []
        for i in range(self.model.nbody):
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i).startswith('B') is True:
                cloth_id.append(i)

        # Material type
        # Mass properties: src/gym_training/envs/mesh/textile_properties.csv
        # Stiffness and damping properties: src/gym_training/envs/mesh/ur5.xml
        materials = 7
        self.model.skin_matid = random.choice(materials)
        if self.model.skin_matid == 7: # Textile 2 (denim)
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 2.29630627176258e-05
            self.model.jnt_stiffness[min(cloth_id):1+max(cloth_id)] = 0.00001
            self.model.dof_damping[min(cloth_id):1+max(cloth_id)] = 0.00001