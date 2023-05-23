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

def add_file_to_gitattributes(filename):
    gitattributes_path = '.gitattributes'
    with open(gitattributes_path, 'a') as file:
        file.write(filename + ' filter=lfs diff=lfs merge=lfs -text\n')

def print_config(config, i=0, string=None):
    if string == None:
        string_config = ''
    else:
        string_config = string
    for x in config:
        x_modified = x.replace('_', '\\_')
        if x in config and config[x] is None:
            string_config += ('{ \phantom - } ' * i + x_modified + ':' + str(config[x]) + '\\\\' + '\n')
        elif isinstance(config[x], int) or isinstance(config[x], float):
            string_config += ('{ \phantom - } ' * i + x_modified + ':' + str(config[x]) + '\\\\' + '\n')
        elif isinstance(config[x], GaussianNoise):
            string_config += ('{ \phantom - } ' * i + x_modified + ':' + 'skrl.resources.noises.torch.gaussian.GaussianNoise' + '\\\\' + '\n')
        elif isinstance(config[x], str):
            string_config += ('{ \phantom - } ' * i + x_modified + ':' + str(config[x]) + '\\\\' + '\n')
        else:
            if x_modified:
                string_config += ('{ \phantom - } ' * i + x_modified + ':' + '\\\\' + '\n')
            string_config = print_config(config[x], i=i + 1, string=string_config)
    
    return string_config

def write_txt_file_from_str(str, file_path=None):
    with open(file_path, 'w') as file:
        file.write(str)

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

class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version

#env = gym.vector.make("InvertedPendulum-v4", num_envs=3, asynchronous=True)
env = gym.vector.make("UR5_ddpg_no_noise", num_envs=3, asynchronous=True)

env = wrap_env(env)

# See the used grafics card
device = torch.cuda.current_device()
print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")
#env.device  ='cpu'
device = env.device

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total)
print("Free memory:", info.free)
print("Used memory:", info.used)

nvidia_smi.nvmlShutdown()

# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=device, replacement=False, export=True)

# Instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#spaces-and-models
models_ddpg = {}
models_ddpg["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models_ddpg["critic"] = DeterministicCritic(env.observation_space, env.action_space, device)
models_ddpg["target_critic"] = DeterministicCritic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_ddpg.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ddpg.html#configuration-and-hyperparameters
cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
cfg_ddpg["exploration"]["noise"] = GaussianNoise(mean=0, std=20, device=device)
cfg_ddpg["exploration"]["timesteps"] = 10000
cfg_ddpg["batch_size"] = 10
cfg_ddpg["random_timesteps"] = 0
cfg_ddpg["learning_starts"] = 10
cfg_ddpg["actor_learning_rate"] = 1e-7
cfg_ddpg["critic_learning_rate"] = 1e-7
# logging to TensorBoard and write checkpoints each 1000 and 1000 timesteps respectively
cfg_ddpg["experiment"]["write_interval"] = 21
cfg_ddpg["experiment"]["checkpoint_interval"] = 500
cfg_ddpg["experiment"]["directory"] = 'runs_for_report'
cfg_ddpg["experiment"]["experiment_name"] = 'DDPG_env_critLR1e-2'
#cfg_ddpg["experiment"]["experiment_name"] = 'InvertedPendulum-v4_test_config_1'

dir = cfg_ddpg["experiment"]["directory"] + '/' + cfg_ddpg["experiment"]["experiment_name"]
print(dir)

directory = './' + cfg_ddpg["experiment"]["directory"]
experiment_name = cfg_ddpg["experiment"]["experiment_name"] + "_config_"
cfg_ddpg["experiment"]["experiment_name"] = experiment_name
experiment_name = cfg_ddpg["experiment"]["experiment_name"]
path_experiment = os.path.join(directory, experiment_name)

# Run the git pull command
repo_name = 'p8_sewbot'

# Find the repository path
current_path = os.getcwd()
repository_path = None

# Iterate through parent directories until the repository is found
while current_path != '/':
    if os.path.isdir(os.path.join(current_path, repo_name, '.git')):
        repository_path = os.path.join(current_path, repo_name)
        break
    current_path = os.path.dirname(current_path)

# Check if the repository path was found
if repository_path:
    print("Repository path:", repository_path)
else:
    print("Repository not found.")

git_pull=True
if git_pull:
    try:
        subprocess.check_output(['git', 'pull'], cwd=repository_path)
        print("Commands executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e.output.decode())
        sys.exit(1)

if not os.path.exists(path_experiment):
    path_experiment = os.path.join(directory, experiment_name + str(1))
    cfg_ddpg["experiment"]["experiment_name"] = experiment_name + str(1)

if os.path.exists(path_experiment):
    print('Experiment already exists. Adding integer identifier.')
    identifier = 1
    while os.path.exists(path_experiment):
        path_experiment = os.path.join(directory, experiment_name + str(identifier))
        cfg_ddpg["experiment"]["experiment_name"] = experiment_name + str(identifier)
        identifier += 1

output_file = os.path.join(path_experiment, "skrl_config.txt")

agent_ddpg = DDPG(models=models_ddpg,
                  memory=memory,
                  cfg=cfg_ddpg,
                  observation_space=env.observation_space,
                  action_space=env.action_space,
                  device=device)

# load checkpoint
inferrence = False
if inferrence:
    agent_ddpg.load("./runs_for_report/DDPG_env_iteration_1_config_1/checkpoints/best_agent.pt")

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 10000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_ddpg)

write_and_push=True

if write_and_push:
    # Write some files with experiment description
    string = print_config(config=cfg_ddpg)

    write_txt_file_from_str(str=string, file_path=output_file)

    output_file = os.path.join(path_experiment, "run_config_csvfile.csv")

    with open(output_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, cfg_ddpg.keys())
        w.writeheader()
        w.writerow(cfg_ddpg)

    filename_to_add = os.path.join(path_experiment, 'checkpoints/best_agent.pt')
    filename_to_add = filename_to_add.replace("./", "")
    add_file_to_gitattributes(filename_to_add)

    # Auto git push the new expiriment to git repo
    try:
        subprocess.check_output(['git', 'lfs', 'install'], stderr=subprocess.STDOUT, cwd=repository_path)
        subprocess.check_output(['git', 'add', '.gitattributes'], cwd=repository_path)
        subprocess.check_output(['git', 'commit', '-m', 'Add .gitattributes file'], cwd=repository_path)
        subprocess.check_output(['git', 'add', output_file], cwd=repository_path)
        subprocess.check_output(['git', 'commit', '-m', 'Add .gitattributes file'], cwd=repository_path)
        subprocess.check_output(['git', 'push'], cwd=repository_path)
        print("Commands executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Command execution failed:", e.output.decode())
        sys.exit(1)

# start training
trainer.train()