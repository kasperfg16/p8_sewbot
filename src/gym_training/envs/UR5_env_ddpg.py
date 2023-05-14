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


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 2.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
    "elevation": -2.0,
}

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
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )
        
        # Action space (In this case - joint limits)
        self.act_space_low = np.array([-np.pi,
                        -np.pi/2,
                        -np.pi,
                        -np.pi,
                        -np.pi,
                        -np.pi,
                        -0.008])
        
        self.act_space_high = np.array([np.pi,
                         np.pi/2,
                         np.pi,
                         np.pi,
                         np.pi,
                         np.pi,
                         0])
        
        self.renderer = mujoco.Renderer(model=self.model)
        self._reset_noise_scale=0.01
        self.action_space = spaces.Box(low=self.act_space_low, high=self.act_space_high, shape=(7,), seed=42, dtype=np.float16)
        self.controller = MJ_Controller(model=self.model, data=self.data, mujoco_renderer=self.mujoco_renderer)
        self.step_counter = 0
        self.graspcompleter = False # to define if a grasp have been made or not. When true, call reward
        # filename = "groundtruth.png"
        # search_path = "./"
        #self.im_background = np.asarray(cv2.imread(find_file(filename, search_path)), np.uint8)
        self.stepcount = 0
        self.goalcoverage = False
        self.area_stack = [0]*2
        self.result_move = False
        self.controller.show_model_info()
        self.home_pose = [np.pi/2, 0, np.pi/2, 0, np.pi/2, 0, 0, 0]
        self.image = np.empty((480, 480, 3)) # image size
        self.in_home_pose = False

        # How many steps are the robot allowed to take before reward?
        self.max_steps = 0

        # Show renders?
        self.headless_mode = False

        # Print output in terminal?
        self.quiet = False

    def step(self, action):

        # init
        self.terminated = False
        self.truncated = False
        self.reward = 0
        self.info = {}
        
        # Stay and let cloth fall
        self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode, tolerance=0.02)
        self.controller.stay(2000, render=not self.headless_mode)
        
        # Check if cloth is already in good position
        self.check()

        # Perform 'self.max_steps' movements based on action guesses from agent and then move to home pose
        self.unfold(action)

        # Compute reward after operation
        self.compute_reward()

        self.step_counter += 1

        # Recive observation after action - Before taking next action
        self.observation  = self._get_obs()

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def unfold(self, action):
        if not self.step_counter >= self.max_steps:
            self.result_move = self.controller.move_group_to_joint_target(target=action, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(10, render=not self.headless_mode)
        else:
            ####### Test
            self.test_grip()
            #######

            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode)
            self.controller.stay(10, render=not self.headless_mode)
            self.truncated = True
            self.step_counter = -1
            self.in_home_pose = True
        

    def _get_obs(self):
        obs = self.image

        # # Resize img to be in observation space (For memory-efficiency)
        new_size = (self.img_width, self.img_width)
        obs = cv2.resize(obs, new_size)

        if not self.headless_mode:
            cv2.imshow('Img used in observation space', obs)
            cv2.waitKey(1)

        # Convert to a more memory-efficient format
        obs = obs.astype(np.uint8)

        return obs

    def truncate(self):

        return self.goalcoverage 

    def check_collision(self):
        # Define when coollision occurs

        return False # bool for collision or not
    
    def check(self):
        self.get_coverage(show=True)
        if self.goalcoverage:
            self.goalcoverage = False
            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet, render=not self.headless_mode, tolerance=0.02)
            self.controller.stay(1000, render=not self.headless_mode)
            self.terminated = True


    def get_coverage(self, show=True):
        #clotharea = 42483.0
        w1 = 10 
        max_stack_size = 2 
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")
        self.image = image.astype(dtype=np.uint8)

        ## use area from ground truth      
        if self.model.skin_matid == 7: # denim
            blur_img = cv2.GaussianBlur(self.image, (9, 9), 3)

        edged = cv2.Canny(self.image, 10, 250)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new2 = cv2.drawContours(self.image, contours, -1, (0,255,0), 3)
            
        if np.size(contours)>0:
            currentarea = cv2.contourArea(contours[0])
            self.area_stack.append(currentarea)
            ## Determine if contour is rectangular
            peri = cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], 0.04 * peri, True)
            # compute the bounding box of the contour and use the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        
            if shape == "square" or shape == "rectangle" and currentarea>10000:
                print("Yay, the cloth is unfolded")
                self.goalcoverage = True
                self.area_stack = [0]*2
        else: self.area_stack.append(0)

        print("area stack after: ", self.area_stack)

        if len(self.area_stack) > max_stack_size:
            self.area_stack.pop(0)

        coveragereward =  w1 * (self.area_stack[1] - self.area_stack[0])
        print("reward", coveragereward)

        return coveragereward

    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self):# Define all the rewards possible           
        coveragereward = self.get_coverage(show=True) # output percentage
        # move complete reward (also acts as a contact/collision penalty)
        if self.result_move == 'success':
            move_complete_reward = 1
        else:
            move_complete_reward = -1

        if not self.quiet:
            print('coveragereward: ', coveragereward)
            print('move complete reward: ', move_complete_reward)

        # Summarize all rewards 
        self.reward = move_complete_reward + coveragereward

    def reset_model(self):
        #self.area_stack = [0]
        self.randomizationSparse() 
        # time.sleep(5)
        self.get_coverage()
        observation = self._get_obs()

        # Initialize manipulator in home pose
        self.data.qpos[self.controller.actuated_joint_ids] = self.home_pose

        return observation

    def test_grip(self):
        til_1 = 15
        tilt2 = 10
        self.controller.open_gripper()
        target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(100, render=not self.headless_mode)

        til_1 = 29
        tilt2 = 15.2
        target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2, np.deg2rad(30), 0, 0]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode)
        self.controller.stay(1000, render=not self.headless_mode)

        self.controller.close_gripper()
        self.controller.stay(1000, render=not self.headless_mode)

        til_1 = 18
        tilt2 = 5
        target = [np.deg2rad(45-185), np.deg2rad(-tilt2), np.deg2rad(90+til_1), np.deg2rad(til_1+tilt2), np.pi/2]
        self.result_move = self.controller.move_group_to_joint_target(target=target, quiet=self.quiet, render=not self.headless_mode, group="Arm")
        self.controller.stay(1000, render=not self.headless_mode)
    
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
        cloth_pertur = 0.15

        # Material type
        materials = range(7, 11, 1)
        self.model.skin_matid = random.choice(materials)
        if self.model.skin_matid == 7: # denim
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 0.004
            #self.model.geom_friction[min(cloth_id):1+max(cloth_id)] =
        elif self.model.skin_matid == 8: # white 1 /
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 0.005
        elif self.model.skin_matid == 9: # white 2 /
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 0.006
        elif self.model.skin_matid == 10: # white 4 / 100% cotton
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 0.007
        elif self.model.skin_matid == 11: # black /
            self.model.body_mass[min(cloth_id):1+max(cloth_id)] = 0.008

        # Cloth mass 
        mass_eps = self.model.body_mass[min(cloth_id)] * cloth_pertur
        self.model.body_mass[min(cloth_id):1+max(cloth_id)] = self.model.body_mass[min(cloth_id)] + random.uniform(-mass_eps, mass_eps)
        # Friction
        frict_eps = self.model.geom_friction[min(cloth_id)] * cloth_pertur
        for i in range(len(self.model.geom_friction[min(cloth_id):1+max(cloth_id)])):
            self.model.geom_friction[i, :] = self.model.geom_friction[i, :] + random.uniform(-frict_eps, frict_eps)
        # Stiffness
        stiff_eps = self.model.jnt_stiffness[min(cloth_id)] * cloth_pertur
        self.model.jnt_stiffness[min(cloth_id):1+max(cloth_id)] = self.model.jnt_stiffness[min(cloth_id)] + random.uniform(-stiff_eps, stiff_eps)
        # Damping
        damp_eps = self.model.dof_damping[min(cloth_id)] * cloth_pertur
        self.model.dof_damping[min(cloth_id):1+max(cloth_id)] = self.model.dof_damping[min(cloth_id)] + random.uniform(-damp_eps, damp_eps)

        # cloth_joint_id = []
        # for i in range(self.model.nbody):
        #     if mujoco.mj_id2name(
        #         self.model,
        #         mujoco.mjtObj.mjOBJ_JOINT,
        #         i
        #         ).startswith('J1') or mujoco.mj_id2name(
        #         self.model,
        #         mujoco.mjtObj.mjOBJ_JOINT,
        #         i).startswith('J0') is True:
        #         cloth_joint_id.append(i)

        # print('cloth_joint_id', cloth_joint_id)

        # # Pose
        # self.data.joint(cloth_joint_id).qpos = 1

        """
        Lightning
        """
        light_pos_eps = self.model.light_pos * cloth_pertur
        for i in range(len(self.model.light_pos)):
            self.model.light_pos[i, :] = self.model.light_pos[i, :] + random.uniform(-light_pos_eps, light_pos_eps)

        light_dir_eps = self.model.light_dir * cloth_pertur
        for i in range(len(self.model.light_dir)):
            self.model.light_dir[i, :] = self.model.light_dir[i, :] + random.uniform(-light_dir_eps, light_dir_eps)

        """
        Pose randomization (all bodies)
        """

        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale

        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nv
        # )
        # self.set_state(qpos, qvel)        
