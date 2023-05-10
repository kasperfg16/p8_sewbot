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
import random


action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(8, ), dtype=np.float64)
observation_space = spaces.Box(0, 5, shape=(6,), dtype=np.float64)


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

        self.observation_space = spaces.Box(0.0, 134.0, shape=(6, ), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=self.observation_space,
            #default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )

        self.step_counter = 0
        
        # Action space (In this case joint limits)
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
        self.action_space = spaces.Box(low=low, high=high, shape=(7,), seed=42)
        self.observation_space = spaces.Box(0, 255, shape=(6,), dtype=np.int64)
        self.controller = MJ_Controller(model=self.model, data=self.data, mujoco_renderer=self.mujoco_renderer)
        self.step_counter = 0
        self.max_steps = 1
        self.graspcompleter = False # to define if a grasp have been made or not. When true, call reward
        filename = "groundtruth.png"
        search_path = "./"
        self.im_background = np.asarray(cv2.imread(find_file(filename, search_path)), np.uint8)
        self.stepcount = 0
        self.goalcoverage = False
        self.area_stack = [0]*2
        self.result_move = False
        #self.controller.show_model_info()
        self.home_pose = [np.pi/2, 0, np.pi/2, 0, np.pi/2, 0, 0]
        self.quiet = False
        #self.cam = self.data.c

    def step(self, action):
        # if self.step_counter == 0: # extra function for getting GT
        #     im =self.mujoco_renderer.render("rgb_array", camera_name="RealSense")
        #     cv2.imwrite("groundtruth.png", im )

        # init
        reward = 0
        terminated = False
        truncated = False
        info = {}
        # Randomization test
        
        # Check if cloth is already in good position
        self.get_coverage()
        if self.goalcoverage:
            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet)
            reward = self.compute_reward()
            terminated = True
            return observation, reward, terminated, truncated, info

        # Perform movements based on action guesses from agent
        observation  = self._get_obs()

        self.step_counter += 1

        # Perform 'self.max_steps' movements based on action guesses from agent and then move to home pose
        if not self.step_counter >= self.max_steps:
            self.result_move = self.controller.move_group_to_joint_target(target=action, quiet=self.quiet)
            self.controller.stay(10)
        else:
            self.result_move = self.controller.move_group_to_joint_target(target=self.home_pose, quiet=self.quiet)
            self.controller.stay(10)
            truncated = True
            self.step_counter = 0
        
        ## Compute reward after operation
        reward = self.compute_reward()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        obs = [1,1,1,1,1,1]
        return obs # Concatenate when multiple obs

    def truncate(self):

        return self.goalcoverage 

    def check_collision(self):
        # Define when coollision occurs

        return False # bool for collision or not
    
    def get_coverage(self):
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")  
        np_array = np.array(image).astype(dtype='float32')
        ## use area from ground truth
        clotharea = 14433.5
        w1 = 100 
        w2 = 300

        new = cv2.subtract(self.im_background, np_array, dtype=cv2.CV_32F)
        imgray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
        imgrayCopy = np.uint8(imgray)
        edged = cv2.Canny(imgrayCopy, 100, 250)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        new2 = cv2.drawContours(np_array, contours, -1, (0,255,0), 3)

        # cv2.imshow("window", image)
        # cv2.waitKey(1)

        currentarea = cv2.contourArea(contours[0])
        self.area_stack.insert(0, currentarea)
        ## compare with ground truth and previous area
        coverageper =  currentarea/clotharea
        print(currentarea)
        coveragereward =  w1 * coverageper + w2 * (self.area_stack[1] - self.area_stack[0])/clotharea 

        if coverageper > 0.9:
            self.goalcoverage = True

        ## Depth testing
        # depth = self.mujoco_renderer.render("depth_array", camera_name="RealSense")  
        # zfar  = 10
        # znear = 0.001
        # depth_linear = (znear * zfar) / (zfar + depth * (znear - zfar))
        # cv2.imshow("window", depth_linear/np.max(depth_linear))
        # cv2.waitKey(1)

        return coveragereward

    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self):
        coveragereward = self.get_coverage() 

        # move complete reward (also acts as a contact/collision penalty)
        if self.result_move == 'success':
            move_complete_reward = 1
            print('move_complete_reward')
        else:
            move_complete_reward = -1

        if not self.quiet:
            print('coveragereward: ', coveragereward)
            print('move complete reward: ', coveragereward)

        # Summarize all rewards 
        total_reward = move_complete_reward + coveragereward

        return total_reward

    def reset_model(self):
        self.randomizationSparse()
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

        """
        Manipulator randomization:
            Initial position of manipulator
            Mechanical properties of the manipulator - what we assume is true +- a small tolerance
        """
        # Initial pos

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
        Perturbation of camera orientation and position
        """
        cam_pos_eps = self.model.cam_pos * cloth_pertur
        for i in range(len(self.model.cam_pos)):
            self.model.cam_pos[i, :] = self.model.cam_pos[i, :] + random.uniform(-cam_pos_eps, cam_pos_eps)        

    def randomizationDense(self): # Randomization between each step
        
        pass