import numpy as np
import gymnasium as gym
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle

action_space = gym.spaces.Discrete(2)
observation_space = gym.spaces.Discrete(2)

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

class UR3Env(gym.Env):
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
    "render_fps": 67,
    }

    def __init__(
        self,
        model = 'src/gym_training/envs/mesh/ur3e.xml',
        reset_noise_scale = 1e-2,
        **kwargs
    ):
        
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        #self.action_space = spaces.Discrete(2, seed=42)
        
        MujocoEnv.__init__(
            self,
            model,
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs
        )
        
        EzPickle.__init__(self, reset_noise_scale, **kwargs)

        self._reset_noise_Scale = reset_noise_scale

    def step(self, action):
        # function for computing position 
        # pos_before = compute_jointpos(self.model, self.data)
        # self.do_simulation(action, self.frame_skip)
        # pos_after = compute_jointpos(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        ctrl_cost = self.control_cost(action)
        obs = observation_space.sample()
        r = 0
        done = False
        return obs, r, done, {}

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _set_action_space(self):
        return super()._set_action_space()
    
    def compute_reward():# input i and i+1 state 
        return None #return reward

    def reset_model(self, *, seed: int or None = None):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
