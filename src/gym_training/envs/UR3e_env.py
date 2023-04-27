import numpy as np
import gymnasium as gym
import mujoco as mj
#reload(gym)
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

action_space = gym.spaces.Discrete(2)
observation_space = gym.spaces.Discrete(2)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
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

class UR3Env(MujocoEnv, EzPickle):
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
        filename = "ur3e.xml"
        search_path = "./"
        model_path = find_file(filename, search_path)
        if model_path is not None:
            print(f"Found {filename} at {model_path}")
        else:
            print(f"{filename} not found in {search_path}")

        MujocoEnv.__init__(
            self,
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            # **kwargs,
        )

        self.observation_space = spaces.Box(0, 5, shape=(6,), dtype=np.float64)
        #self.action_space = spaces.Discrete(2, seed=42)
        
    def step(self, action):
        # function for computing position 
        obs = self._get_obs()
        self.do_simulation(action, self.frame_skip)
        observation  = self._get_obs()
        
        ### Compute reward
        #reward = self.compute_reward()

        ### Check if need for more training
            ## Collision, succes, max. time steps
        #done = self.check_collision()

        #self.render_mode = "human"
        #self.render()
        
        ## Create an Image object from the array
        #self.render_mode = "r"
        #self.render()
        #img = Image.fromarray(np_arr)

        #plt.imshow(img)
        #plt.show()
        reward = self.compute_reward()
        terminated = False # Check for collision or success
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info 

    def _get_obs(self):
        joint_pos = self.data.qpos.flat.copy()
        self.render_mode = "rgb_array"
        image = self.render()

        return joint_pos #, image # Concatenate when multiple obs

    
    def check_collision():
        # Define when coollision occurs
        return False # bool for collision or not
    
    def _set_action_space(self):
        # Define a set of actions to execute in the simulation
        return super()._set_action_space()
    
    def compute_reward(self):# Define all the rewards possible
        # Grasp reward 1 for open, 0 for close
        # 
        # Coverage reward
        # if >90% coverage, call terminate
        # Reward for standing in a certain position
        goal_pos = [0, -1.5, 1.5, 0, 0, 0] # Home pos ish
        position = self.data.qpos.flat.copy()
        error_pos = sum( abs(np.subtract(goal_pos, position)))
        # Summarize all rewards 
        total_reward = 10 - error_pos

        return total_reward #return reward

    def reset_model(self, *, seed: int or None = None):
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nv
        # )
        # self.set_state(qpos, qvel)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()

        observation = self._get_obs()

        return observation
