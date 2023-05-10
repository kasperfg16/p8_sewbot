import numpy as np
import gymnasium as gym
#from gym_training.controller.UR3e_contr import UR3e_controller
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
import sys


# DEFAULT_CAMERA_CONFIG = {
#     "trackbodyid": 1,
#     "distance": 4.0,
#     "lookat": np.array((0.0, 0.0, 2.0)),
#     "elevation": -20.0,
# }

def find_file(filename, search_path):
    """
    Search for a file in the given search path.
    Returns the full path to the file if found, or None otherwise.
    """
    
    for root, dir, files in os.walk(search_path):
        if filename in files:
            return os.path.abspath(os.path.join(root, filename))
    return None

class UR5Env_ddpg(MujocoEnv, EzPickle):
    """
    ### Action space
    | Num | Action                | Control Min | Control Max | Name in XML file | Joint | Unit |
    |-----|-----------------------|-------------|-------------|------------------|-------|------|
    | 1   | Joint pos             |
    | 2   | Joint pos             |
    | 3   | Joint pos             |
    | 4   | Joint pos             |
    | 5   | Joint pos             |
    | 6   | Joint pos             |
    | 7   | Open or close gripper |



    ### Observation Space
    | Num | Observation             | Min  | Max | Name in XML file | Joint | Unit      |
    | --- | ----------------------  | ---- | --- | -----------------| ----- | ----------|
    | 1   | Joint pos               |
    | 2   | Joint pos               |
    | 3   | Joint pos               |
    | 4   | Joint pos               |
    | 5   | Joint pos               |
    | 6   | Joint pos               |
    | 7   | Open or close gripper   |
    | 8   | Intersec. over union    |
    | 9   | Surroundings/collision

    ### Rewards & Penalties
    | Num | Name           | Descroption               | Dense/Sparse| Point |
    | --- | ---------------|---------------------------|-------------|-------|
    |  1  | Succes reward  | Getting 90-100% coverage  | Dense       | 100   |
    |  2  | Step penalty   | Getting penalty each step | Sparse      | -10   |
    |  3  | Collision      | Collision with table      | Dense       | -1000 |
    |  4  | Grasp success  | Grasping the cloth        | Sparse      | 10

    """

    metadata = {
        "render_modes": [
        "human",
        "rgb_array",
        "depth_array",
    ],
    "render_fps": 100,
    }
    
    def __init__(
        self,
        **kwargs
    ):
        EzPickle.__init__(self,  **kwargs)
        filename = "ur5.xml"
        search_path = "./"
        model_path = find_file(filename, search_path)
        # if model_path is not None:
        #     print(f"Found {filename} at {model_path}")
        # else:
        #     print(f"{filename} not found in {search_path}")

        self.img_width = 100
        self.img_height = 100
        self.observation_space = spaces.Box(0, 255, shape=(self.img_width, self.img_height, 3), dtype=np.uint8)

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=self.observation_space,
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )
        
        # Action space (In this case - joint limits)
        low = np.array([-np.pi,
                        -np.pi/2,
                        -np.pi,
                        -np.pi,
                        -np.pi,
                        -np.pi,
                        -0.008])
        
        high = np.array([np.pi,
                         np.pi/2,
                         np.pi,
                         np.pi,
                         np.pi,
                         np.pi,
                         0])
        
        self.renderer = mujoco.Renderer(model=self.model)
        self._reset_noise_scale=0
        self.action_space = spaces.Box(low=low, high=high, shape=(7,), seed=42, dtype=np.float16)
        self.controller = MJ_Controller(model=self.model, data=self.data, mujoco_renderer=self.mujoco_renderer)
        self.step_counter = 0
        self.graspcompleter = False # to define if a grasp have been made or not. When true, call reward
        filename = "groundtruth.png"
        search_path = "./"
        self.im_background = np.asarray(cv2.imread(find_file(filename, search_path)), np.uint8)
        self.stepcount = 0
        self.goalcoverage = False
        self.area_stack = [0]*2
        self.result_move = False
        self.controller.show_model_info()
        self.home_pose = [np.pi/2, 0, np.pi/2, 0, np.pi/2, 0, 0]
        self.image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")
        self.in_home_pose = False

        # How many steps are the robot allowed to take before reward?
        self.max_steps = 0

        # Show renders?
        self.headless_mode = False

        # Print output in terminal?
        self.quiet = False

    def step(self, action):

        # init
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # Check if cloth is already in good position
        self.get_coverage(show=False)
        if self.goalcoverage:
            self.goalcoverage = False
            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            reward = self.compute_reward()
            observation  = self._get_obs()
            terminated = True
            return observation, reward, terminated, truncated, info

        # Perform 'self.max_steps' movements based on action guesses from agent and then move to home pose
        if not self.step_counter >= self.max_steps:
            self.result_move = self.controller.move_group_to_joint_target(target=action, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(10, render=not self.headless_mode)
        else:
            ####### Test
            til_1 = 15
            tilt2 = 10
            target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2]
            self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
            self.controller.stay(100, render=not self.headless_mode)

            til_1 = 28
            tilt2 = 15
            target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2, np.deg2rad(30), 0, 0]
            self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(1000, render=not self.headless_mode)

            self.controller.close_gripper()
            self.controller.stay(1000, render=not self.headless_mode)

            til_1 = 20
            tilt2 = 5
            target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2]
            self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")

            self.controller.stay(1000, render=not self.headless_mode)

            #######

            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(10, render=not self.headless_mode)
            truncated = True
            self.step_counter = -1
            self.in_home_pose = True
        
        # Compute reward after operation
        reward = self.compute_reward()

        # Recive observation that is send to the NN/agent
        observation  = self._get_obs()

        self.step_counter += 1

        print('tilløk')

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = self.image

        # Resize img to be in observation space (For memory-efficiency)
        new_size = (self.img_width, self.img_width)
        resized_image = cv2.resize(obs, new_size)

        if not self.headless_mode:
            cv2.imshow('Img used in observation space', resized_image)
            cv2.waitKey(1)

        # Convert to a more memory-efficient format
        obs = resized_image.astype(np.uint8)

        return obs # Concatenate when multiple obs

    def truncate(self):

        return self.goalcoverage 

    def check_collision(self):
        # Define when coollision occurs

        return False # bool for collision or not
    
    def get_coverage(self, show=True):
        self.image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")  
        np_array = np.array(self.image).astype(dtype=np.uint8)
        ## use area from ground truth
        clotharea = 14433.5
        w1 = 100 
        w2 = 300
        max_stack_size = 5

        new = cv2.subtract(self.im_background, np_array)
        imgray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        imgrayCopy = np.uint8(imgray)
        edged = cv2.Canny(imgrayCopy, 100, 250)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new2 = cv2.drawContours(np_array, contours, -1, (0,255,0), 3)

        currentarea = cv2.contourArea(contours[0])
        self.area_stack.insert(0, currentarea)
        ## compare with ground truth and previous area
        coverageper =  currentarea/clotharea

        if show and not self.headless_mode:
            cv2.imshow("Camera", new2)
            cv2.waitKey(1)
        
        if len(self.area_stack) > max_stack_size:
            self.area_stack.pop(0)

        coveragereward =  w1 * coverageper + w2 * (self.area_stack[1] - self.area_stack[0])/clotharea 

        if coverageper > 0.9:
            self.goalcoverage = True

        return coveragereward

    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self):# Define all the rewards possible
        # Grasp reward 1 for open, 0 for close
        ## Coverage reward if >90% coverage, call terminate
        ## Compute only coverage after a grasp - remember to change   
             
        coveragereward = self.get_coverage(show=self.in_home_pose) # output percentage

        # move complete reward (also acts as a contact/collision penalty)
        if self.result_move == 'success':
            move_complete_reward = 1
        else:
            move_complete_reward = -1

        if not self.quiet:
            print('coveragereward: ', coveragereward)
            print('move complete reward: ', move_complete_reward)

        # Summarize all rewards 
        total_reward = move_complete_reward + coveragereward

        return total_reward

    def reset_model(self):
        self.in_home_pose = False

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        self.img_stack=[]

        observation = self._get_obs()

        return observation