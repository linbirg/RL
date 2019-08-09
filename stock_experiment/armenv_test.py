from arm_env import ArmEnv

MODE = ['easy', 'hard']
n_model = 1

env = ArmEnv(mode=MODE[n_model])

env = ArmEnv(mode=MODE[n_model])
S_DIM = env.state_dim
A_DIM = env.action_dim
A_BOUND = env.action_bound[1]

print(S_DIM)
print(A_DIM)
print(A_BOUND)
s = env.reset()
print(s)
a = env.sample_action()
print(a)
r = env.step(a)
print(r)