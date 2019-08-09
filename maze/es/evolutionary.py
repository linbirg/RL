"""
Simple code for Distributed ES proposed by OpenAI.
Based on this paper: Evolution Strategies as a Scalable Alternative to Reinforcement Learning
Details can be found in : https://arxiv.org/abs/1703.03864
Visit more on my tutorial site: https://morvanzhou.github.io/tutorials/
"""
import os
import time
import multiprocessing as mp
import numpy as np
# import gym

from maze_env import MazeEnv
from maze import Maze

import logger

logger = logger.Logger(show_in_console=False)

N_KID = 15  # half of the training population
N_GENERATION = 100000  # training step
LR = .001  # learning rate
SIGMA = .005  # mutation strength or step size
N_CORE = mp.cpu_count() - 1

PARAMS_FILE_NAME = "params.npy"

CONFIG = [
    dict(
        game="CartPole-v0",
        n_feature=4,
        n_action=2,
        continuous_a=[False],
        ep_max_step=700,
        eval_threshold=500),
    dict(
        game="MountainCar-v0",
        n_feature=2,
        n_action=3,
        continuous_a=[False],
        ep_max_step=200,
        eval_threshold=-120),
    dict(
        game="Pendulum-v0",
        n_feature=3,
        n_action=1,
        continuous_a=[True, 2.],
        ep_max_step=200,
        eval_threshold=-180),
    dict(
        game="maze",
        n_feature=2,
        n_action=4,
        continuous_a=[False],
        ep_max_step=1000,
        eval_threshold=2000)
][3]  # choose your game

# params_rnd = [[0]] * N_KID * 2


def sign(k_id):
    return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class ParamsRand(object):
    def __init__(self, shape):
        self.shape = shape
        self.params = [[0]] * shape[0]

    def rand(self):
        for i in range(len(self.params)):
            self.params[i] = dropout(np.random.randn(self.shape[1]), 0.8)


class SGD(object):  # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

    def save(self, file="sgd"):
        np.save(file, self.v)

    def load(self, file="sgd"):
        if os.path.exists(file):
            self.v = np.load(file)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# 一维数组
def dropout(params, rate=0.5):
    if rate < 0. or rate >= 1:
        raise Exception('Dropout level must be in interval [0, 1]')
    retain_prob = 1 - rate
    # 我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    # 硬币 正面的概率为p，n表示每个神经元试验的次数
    # 因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    # 即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    sample = np.random.binomial(n=1, p=retain_prob, size=params.shape)
    params *= sample
    params /= retain_prob
    return params


def params_reshape(shapes, params):  # reshape to be a matrix
    p, start = [], 0
    for _, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [
            params[start:start + n_w].reshape(shape),
            params[start + n_w:start + n_w + n_b].reshape((1, shape[1]))
        ]
        start += n_w + n_b
    return p


def get_reward(
        shapes,
        params,
        env,
        ep_max_step,
        rand_param_obj,
        seed_and_id=None,
):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        # params += sign(k_id) * SIGMA * np.random.randn(params.size)
        params += sign(k_id) * SIGMA * rand_param_obj.params[k_id]
        # logger.debug(rand_param_obj.params)
    p = params_reshape(shapes, params)
    # run episode
    # s = env.reset()
    my_env = MazeEnv(env.maze)
    my_env.maze.door = (env.maze.door[0], env.maze.door[1])
    start = my_env.reset()
    # start = my_env.get_state()
    # logger.debug(["kid ",seed_and_id[1] if seed_and_id is not None else None,"door",my_env.maze.door,"env.door",env.maze.door,"start",start])
    ep_r = 0.
    for g in range(ep_max_step):
        s = my_env.get_state()
        a = get_action(p, s)
        s, r, done = my_env.step(a)
        ep_r += r
        if done:
            break
    return ep_r


def get_action(params, x):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    p_weights = softmax(x[0])
    a = np.random.choice(range(x.shape[1]), p=p_weights)
    # a = np.argmax(x, axis=1)[0]
    return a


def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))

    # 2*200,200*50,50*4
    s0, p0 = linear(CONFIG['n_feature'], 400)
    s1, p1 = linear(400, 100)
    s2, p2 = linear(100, CONFIG['n_action'])
    return [s0, s1, s2], np.concatenate((p0, p1, p2))


def train(net_shapes, net_params, optimizer, utility, pool, rand_param_obj):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(
        0, 2**32 - 1, size=N_KID,
        dtype=np.uint32).repeat(2)  # mirrored sampling

    env.reset()
    rand_param_obj.rand()
    # distribute training in parallel
    jobs = [
        pool.apply_async(get_reward, (
            net_shapes,
            net_params,
            env,
            CONFIG['ep_max_step'],
            rand_param_obj,
            [noise_seed[k_id], k_id],
        )) for k_id in range(N_KID * 2)
    ]
    rewards = np.array([j.get() for j in jobs])
    kids_rank = np.argsort(rewards)[::-1]  # rank kid id by reward

    cumulative_update = np.zeros_like(net_params)  # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])  # reconstruct noise using seed
        # cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)
        cumulative_update += utility[ui] * sign(k_id) * rand_param_obj.params[
            k_id]
    # logger.debug(rand_param_obj.params)
    gradients = optimizer.get_gradients(cumulative_update /
                                        (2 * N_KID * SIGMA))
    return net_params + gradients, rewards


def gen_utility(kids_cnt):
    base = kids_cnt * 2  # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    new_utility = util_ / util_.sum() - 1 / base
    return new_utility


def try_work(env, ep_max_step, net_param):
    s = env.reset()
    done = False
    for step in range(ep_max_step):
        env.render()
        a = get_action(net_param, s)
        s, r, done = env.step(a)
        logger.debug([a, r, s, done, step])
        if done:
            break
        # time.sleep(0.02)
    return done


if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    # base = N_KID * 2  # *2 for mirrored samplingsampling
    # rank = np.arange(1, base + 1)
    # util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    # utility = util_ / util_.sum() - 1 / base
    utility = gen_utility(N_KID)

    # training
    net_shapes, net_params = build_net()
    if os.path.exists(PARAMS_FILE_NAME):
        net_params = np.load(PARAMS_FILE_NAME)
    # env = gym.make(CONFIG['game']).unwrapped
    maze = None
    if os.path.exists("maze.json"):
        maze = Maze.reload("maze.json")
    env = MazeEnv(maze=maze)
    env.maze.save()

    optimizer = SGD(net_params, LR)
    optimizer.load()  # 会尝试load一下保存的参数。
    pool = mp.Pool(processes=N_CORE)
    mar = None  # moving average reward
    param_rand_obj = ParamsRand((N_KID * 2, net_params.size))
    for g in range(N_GENERATION):
        t0 = time.time()
        net_params = dropout(net_params, 0.3)  # dropout
        net_params, kid_rewards = train(net_shapes, net_params, optimizer,
                                        utility, pool, param_rand_obj)

        # np.savetxt("params.csv", net_params, fmt="%0.3f", delimiter=',', newline='\n')
        np.save(PARAMS_FILE_NAME, net_params)
        optimizer.save()  # 保存sgd参数
        # test trained net without noise
        # env.reset()
        net_r = get_reward(
            net_shapes,
            net_params,
            env,
            CONFIG['ep_max_step'],
            None,
        )
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r  # moving average reward
        try:
            print(
                'Gen: ',
                g,
                '| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % kid_rewards.mean(),
                '| Kid_max_R: %.1f' % max(kid_rewards),
                '| Gen_T: %.2f' % (time.time() - t0),
            )
            if mar >= CONFIG['eval_threshold']: break
        except Exception as e:
            print(e)
        if g % 5 == 0:
            p = params_reshape(net_shapes, net_params)
            try_work(env, 1000, p)
            env.close()
            env.viewer = None

    # test
    print("\nTESTING....")
    p = params_reshape(net_shapes, net_params)
    while True:
        try_work(env, 2000, p)
