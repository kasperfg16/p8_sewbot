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
        
        # Action space
        low = np.array([-3.14159,
                        -3.14159,
                        -3.14159,
                        -3.14159,
                        -3.14159,
                        -3.14159,
                        -0.008])
        
        high = np.array([3.14159,
                         3.14159,
                         3.14159,
                         3.14159,
                         3.14159,
                         3.14159,
                         0])
        
        self._reset_noise_scale=1
        self.action_space = spaces.Box(low=low, high=high, shape=(7,), seed=42)
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
        self.result_move = False

    def step(self, action):

        self.result_move = self.controller.move_group_to_joint_target(target=action, quiet=True)

        image = self.mujoco_renderer.render("human")

        observation  = self._get_obs()
        
        ## Compute reward only after a pick and place operation
        reward = self.compute_reward()
        terminated = False
        truncated = False
        info = {}

        self.step_counter += 1
        if self.step_counter >= 20:
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
        stack=3
        ## make continuous background subtraction (or something) to keep history of cloth location behind manipulator
        self.img_stack.append(image)
        #print(len(self.img_stack))
        cv2.imshow("image", image)

        if len(self.img_stack)>=stack:
            self.img_stack.pop(0)
            np_array = np.array(self.img_stack)
            sequence = np.median(np_array, axis=0).astype(dtype='float32') # take median filter over the image stack
            new = cv2.subtract(self.im_background, sequence)
            imgray = cv2.cvtColor(new, cv2.COLOR_RGB2GRAY)
            imgrayCopy = np.uint8(imgray)
            edged = cv2.Canny(imgrayCopy, 30, 200)
            contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            new2 = cv2.drawContours(new, contours, -1, (0,255,0), 3)

            cv2.imshow("sequence", sequence)
            cv2.imshow("subtract", new)
            cv2.waitKey(10)

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
    
    def compute_reward(self):# Define all the rewards possible
        # Grasp reward 1 for open, 0 for close
        ## Coverage reward if >90% coverage, call terminate
        ## Compute only coverage after a grasp - remember to change        
        coveragereward = self.get_coverage() # output percentage
        print('coveragereward', coveragereward)

        # Contact/collision penalty
        if self.result_move == 'success':
            move_complete_reward = 10
        else:
            move_complete_reward = -1

        collision = self.check_collision()
        # Reward for standing in a certain position

        # Summarize all rewards 
        total_reward = move_complete_reward + coveragereward

        return total_reward

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        print(self.init_qpos)


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