import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle
from gym_training.controller.UR5_contr import URController
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from gym_training.controller.mujoco_controller import MJ_Controller


##############
### Get camera to work!!!
#############

action_space = gym.spaces.Discrete(2)
observation_space = gym.spaces.Discrete(2)

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

class UR5Env_dqn(MujocoEnv, EzPickle):
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

        self.observation_space = gym.spaces.Box(low=-1.0, high=2.0, shape=(6, ), dtype=np.float32)
        
        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=self.observation_space
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )

        #self.controller = URController()
        #self.controller = MJ_Controller()

        self.step_counter = 0
        self.observation_space = spaces.Box(0.0, 134.0, shape=(138, ), dtype=np.float64)
        self.action_space = spaces.Discrete(6, seed=42)
        #self.controller = UR3e_controller(self.model, self.data, self.render
        self.graspcompleter = False # to define if a grasp have been made or not. When true, call reward
        self.im_background = cv2.imread('/home/marie/Desktop/p8_sewbot/src/gym_training/table.png')
        self.stepcount = 0
        self.img_stack = []*20
        self.goalcoverage = False
        self.area_stack = [0]*2
        
    def step(self, action):
        self.stepcount = self.stepcount +1
        #print(self.stepcount)
        # function for computing position 
        #obs = self._get_obs()
        #print('action', action)
        self.do_simulation(action, self.frame_skip)
        observation  = self._get_obs()

        terminated = False # Check for collision or success
        truncated = False
        info = {}

        # joint_angles = [1, 1, 1, 1, 1]
        # result = self.controller.move_group_to_joint_target(group="Arm", target=joint_angles)

        ### Compute reward
        #reward = self.compute_reward()
        ### Check if need for more training
            ## Collision, succes, max. time steps
        #done = self.check_collision()
        #self.controller.inverse_kinematics()
        # Render scene
        self.render_mode = "human"
        self.render()

        # pos_delta = np.array([0.000001, 0.0, 0])
        # self.data.mocap_pos[:] = self.data.mocap_pos + pos_delta

        #print(self.data.mocap_pos[:])
        # function for computing position 
        #obs = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        observation  = self._get_obs()
        ## Compute reward only after a pick and place operation
        reward = self.compute_reward()
        terminated = False # Check for collision or successs
        truncated = False
        info = {}
        #plt.imshow(img)
        #plt.show()
        self.step_counter += 1
        if self.step_counter >= 20000:
            truncated = True
            self.step_counter = 0

        
        reward = self.compute_reward()
        return observation, reward, terminated, truncated, info 

    def _get_obs(self):
        joint_pos = self.data.qpos[:6].astype(np.float32)
        #print('joint_pos', joint_pos)
        # Define the size of the image

        # !!!Has to be changed!!! Generate a random 3D array of values between 0 and 1
        #img_array = np.random.rand(self.img_width, self.img_height).astype(np.uint8)

        #print(len(self.data.ctrl))
        return joint_pos #, image # Concatenate when multiple obs

    
    def check_collision(self):
        # Define when collision occurs

        return False # bool for collision or not
    
    def get_coverage(self, image):
        # use area from ground truth
        clotharea = 14433.5
        tol = 0.005

        # # make continuous background subtraction (or something) to keep history of cloth location behind manipulator
        # self.img_stack.insert(0, image)
        # sequence = np.median(self.img_stack, axis=0).astype(dtype=np.uint8)
        # new = cv2.subtract(self.im_background, sequence)
        # imgray = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)

        # edged = cv2.Canny(imgray, 30, 200)
        # cv2.waitKey(0)
        # contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # new = cv2.drawContours(new, contours, -1, (0,255,0), 3)
        # currentarea = cv2.contourArea(contours[0])
        # self.area_stack.insert(0, currentarea)
        # ## compare with ground truth and previous area
        # coverageper = (self.area_stack[0] - self.area_stack[1]) + 100 - (clotharea - currentarea)/clotharea
        # #print(coverageper)
        # if (clotharea - currentarea)/clotharea > 0.9:
        #     goalcoverage = True
        # return coverage percentage
        return 0.9
    
    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self):# Define all the rewards possible
        # Grasp reward 1 for open, 0 for close
        # 
        ## Coverage reward if >90% coverage, call terminate
        ## Compute only coverage after a grasp - remember to change
        image = self.mujoco_renderer.render("rgb_array", camera_name="RealSense")         
        coverage = self.get_coverage(image) # output percentage

        # Contact/collision penalty
        collision = self.check_collision()
        # Reward for standing in a certain position
        goal_pos = [0, -1.5, 1.5, 0, 0, 0] # Home pos ish
        position = self.data.qpos[:6]
        error_pos = np.sum( np.abs(np.subtract(goal_pos, position)))
        # Summarize all rewards 
        total_reward = 10 - error_pos

        return total_reward #return reward

    def reset_model(self, *, seed: int or None = None):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()

        observation = self._get_obs()

        return observation
