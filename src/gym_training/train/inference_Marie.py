import csv
import os
import sys
import gymnasium as gym
import gym_training

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
import nvidia_smi
import subprocess


# Define the models (deterministic models) for the DDPG agent using mixin
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return 314 * torch.tanh(self.action_layer(x)), {}  # Pendulum-v1 action_space is -2 to 2



# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version

#env = gym.vector.make("InvertedPendulum-v4", num_envs=3, asynchronous=True)
env = gym.vector.make("UR5_ddpg_touch", num_envs=1, asynchronous=True)

env = wrap_env(env)

# See the used grafics card
device = torch.cuda.current_device()
print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")
#env.device  ='cpu'
device = env.device

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

# Instantiate the agent's policy.
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["random_timesteps"] = 0
# logging to TensorBoard each 300 timesteps and ignore checkpoints
cfg_ddpg["experiment"]["write_interval"] = 300
cfg_ddpg["experiment"]["checkpoint_interval"] = 0

agent_ddpg = DDPG(models=models_ddpg,
                  memory=None,
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)

# load checkpoint

agent_ddpg.load("./Maries_runs/DDPG_touch_v4_config_3/checkpoints/best_agent.pt")



# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

# evaluate the agent
trainer.eval()