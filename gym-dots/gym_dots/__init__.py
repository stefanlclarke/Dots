from gym.envs.registration import register

register(
    id='dots-v0',
    entry_point='gym_dots.envs:DotsEnvironment',
)
