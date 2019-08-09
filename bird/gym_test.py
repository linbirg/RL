import gym

GAME = "MountainCarContinuous-v0"
# GAME = "MountainCarContinuous-v0"
# GAME = "MountainCarContinuous-v0"
env = gym.make(GAME)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
