from gymnasium.envs.registration import register

register(
    id='UR3e',
    entry_point='gym_training.envs:UR3Env'
)

register(
    id='UR5',
    entry_point='gym_training.envs:UR5Env'
)