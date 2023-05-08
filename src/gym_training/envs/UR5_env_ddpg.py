import numpy as np
import gymnasium as gym
#from gym_training.controller.UR3e_contr import UR3e_controller
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
from gymnasium import spaces
from gymnasium.utils import EzPickle
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from gym_training.controller.mujoco_controller import MJ_Controller


action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(8, ), dtype=np.float64)
observation_space = spaces.Box(0, 5, shape=(6,), dtype=np.float64)


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

        
        self.observation_space = spaces.Box(0.0, 134.0, shape=(6, ), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=self.observation_space,
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )

        self.step_counter = 0
        self.action_space = spaces.Box(low=-150, high=150, shape=(7, ), dtype=np.float64)
        self.observation_space = spaces.Box(0, 5, shape=(6,), dtype=np.float64)
        self.controller = MJ_Controller(model=self.model)
        self.step_counter = 0

        self.graspcompleter = False # to define if a grasp have been made or not. When true, call reward
        filename = "table.png"
        search_path = "./"
        self.im_background = np.asarray(cv2.imread(find_file(filename, search_path)), np.float32)
        self.stepcount = 0
        self.img_stack = []#np.zeros([20, 480, 480, 3])
        self.goalcoverage = False
        self.area_stack = [0]*2
        

    def step(self, action):
        #self.stepcount = self.stepcount +1
        # function for computing position 
        #obs = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        observation  = self._get_obs()
        reward = self.get_coverage()
        terminated = False # Check for collision or success
        truncated = self.truncate()
        info = {}

        self.step_counter += 1
        if self.step_counter >= 2000:
            truncated = True
            self.step_counter = 0

        return observation, reward, terminated, truncated, info 

    def _get_obs(self):
        joint_pos = self.data.qpos[:6]
        obs = joint_pos#np.concatenate(joint_pos, image)
        return obs # Concatenate when multiple obs

    def truncate(self):

        return self.goalcoverage 

    def check_collision(self):
        # Define when coollision occurs

        return False # bool for collision or not
    
    def get_coverage(self):

        ## use area from ground truth
        clotharea = 14433.5
        w1 = 100 
        w2 = 300
        stack=10
        ## make continuous background subtraction (or something) to keep history of cloth location behind manipulator
        self.img_stack.append(image)
        #print(len(self.img_stack))

        if len(self.img_stack)==stack:
            self.img_stack.pop(0)
            np_array = np.array(self.img_stack)
            sequence = np.median(np_array, axis=0).astype(dtype='float32') # take median filter over the image stack
            new = cv2.subtract(self.im_background, sequence)
            imgray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
            imgrayCopy = np.uint8(imgray)
            edged = cv2.Canny(imgrayCopy, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            new2 = cv2.drawContours(new, contours, -1, (0,255,0), 3)

            # cv2.imshow("sequence", sequence)
            # cv2.imshow("subtract", new)
            # cv2.waitKey(1)

            currentarea = cv2.contourArea(contours[0])
            self.area_stack.insert(0, currentarea)
            ## compare with ground truth and previous area
            coverageper =  currentarea/clotharea
            coveragereward =  w1 * coverageper + w2 * (self.area_stack[1] - self.area_stack[0])/clotharea 

            if coverageper > 0.9:
                self.goalcoverage = True
        else:
            coveragereward = 0

        return coveragereward

    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self, image):# Define all the rewards possible
        # Grasp reward 1 for open, 0 for close
        # 
        ## Coverage reward if >90% coverage, call terminate
        ## Compute only coverage after a grasp - remember to change       
        coveragereward = self.get_coverage(image) # output percentage

        # Contact/collision penalty
        collision = self.check_collision()
        # Reward for standing in a certain position
        # goal_pos = [0, -1.5, 1.5, 0, 0, 0] # Home pos ish
        # position = self.data.qpos[:6]
        # error_pos = np.sum( np.abs(np.subtract(goal_pos, position)))
        # Summarize all rewards 
        #total_reward = error_pos

        return coveragereward

    def reset_model(self, *, seed: int or None = None):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()

        observation = self._get_obs()

        return observation
