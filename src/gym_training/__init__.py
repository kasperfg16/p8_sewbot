from gymnasium.envs.registration import register

register(
    id='UR3e',
    entry_point='gym_training.envs:UR3Env'
)

register(
    id='UR5_dqn',
    entry_point='gym_training.envs:UR5Env_dqn'
)

register(
    id='UR5_ddpg',
    entry_point='gym_training.envs:UR5Env_ddpg'
)