from gymnasium.envs.registration import register

register(
    id='UR5_dqn',
    entry_point='gym_training.envs:UR5Env_dqn'
)

register(
    id='UR5_ddpg',
    entry_point='gym_training.envs:UR5Env_ddpg'
)

register(
    id='UR5_ddpg_no_noise',
    entry_point='gym_training.envs:UR5Env_ddpg_no_noise'
)

register(
    id='testEnv',
    entry_point='gym_training.envs:test_Env'
)