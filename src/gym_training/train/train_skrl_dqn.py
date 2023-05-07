import gymnasium as gym
import gym_training


# Import the skrl components to build the RL system
from skrl.utils.model_instantiators import deterministic_model, Shape
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
import torch
import gc

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
# Load and wrap the Gymnasium environment.
# Note: the environment version may change depending on the gymnasium version
display = True

if display:
    render_mode = 'human'
else:
    render_mode = None

env = gym.vector.make("Pendulum-v1", num_envs=1, asynchronous=False, render_mode=render_mode)

env = wrap_env(env)

# See the used grafics card
device = torch.cuda.current_device()
#print(f"Using CUDA device {device}: {torch.cuda.get_device_name(device)}")

env.device = 'cpu'
device = env.device

# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=5000, num_envs=env.num_envs, device=device, replacement=False)

# Instantiate the agent's models (function approximators) using the model instantiator utility
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#spaces-and-models
models_dqn = {}
models_dqn["q_network"] = deterministic_model(observation_space=env.observation_space,
                                              action_space=env.action_space,
                                              device=device,
                                              clip_actions=False,
                                              input_shape=Shape.OBSERVATIONS,
                                              hiddens=[64, 64],
                                              hidden_activation=["relu", "relu"],
                                              output_shape=Shape.ACTIONS,
                                              output_activation=None,
                                              output_scale=1.0)
models_dqn["target_q_network"] = deterministic_model(observation_space=env.observation_space,
                                                     action_space=env.action_space,
                                                     device=device,
                                                     clip_actions=False,
                                                     input_shape=Shape.OBSERVATIONS,
                                                     hiddens=[64, 64],
                                                     hidden_activation=["relu", "relu"],
                                                     output_shape=Shape.ACTIONS,
                                                     output_activation=None,
                                                     output_scale=1.0)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_dqn.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#configuration-and-hyperparameters
cfg_dqn = DQN_DEFAULT_CONFIG.copy()
cfg_dqn["learning_starts"] = 100
cfg_dqn["exploration"]["final_epsilon"] = 0.04
cfg_dqn["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_dqn["experiment"]["write_interval"] = 1000
cfg_dqn["experiment"]["checkpoint_interval"] = 5000

agent_dqn = DQN(models=models_dqn,
                memory=memory,
                cfg=cfg_dqn,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 500, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_dqn)

# start training
trainer.train()