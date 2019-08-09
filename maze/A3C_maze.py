"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import time
import tensorflow as tf
import numpy as np

import os
import shutil

import matplotlib.pyplot as plt

# maze
from env.maze_env import MazeEnv
from env.maze import Maze

from ai.A3C import A3CNet

import logger

# GAME = 'CartPole-v0'
# GAME = 'MountainCar-v0'
OUTPUT_GRAPH = True
LOG_DIR = './board-log'
N_WORKERS = multiprocessing.cpu_count()
MAX_TOTAL_STEP = 20000
MAX_GLOBAL_EP = 50000

UPDATE_GLOBAL_ITER = 500

logger = logger.Logger(show_in_console=False)


class Worker(object):
    GAMMA = 0.9
    GLOBAL_RUNNING_R = []
    GLOBAL_EP = 0

    def __init__(self, sess, name, N_S, N_A, globalAC, maze=None):
        self.SESS = sess
        self.N_S = N_S
        self.N_A = N_A
        self.env = MazeEnv(log_name=name, maze=maze)
        self.name = name
        self.AC = A3CNet(self.SESS, self.name, self.N_S, self.N_A, globalAC)
        # self.saver = tf.train.Saver()

    def _record_global_reward_and_print(self, global_runing_rs, ep_r,
                                        global_ep, total_step):
        global_runing_rs.append(ep_r)
        try:
            print(self.name, "Ep:", global_ep,
                  "| Ep_r: %i" % global_runing_rs[-1], "| total step:",
                  total_step)
        except Exception as e:
            print(e)

    def train(self):
        buffer_s, buffer_a, buffer_r = [], [], []
        s = self.env.reset()
        ep_r = 0
        total_step = 1

        def reset():
            nonlocal ep_r, total_step
            self.env.reset()
            ep_r = 0
            total_step = 1

        while not COORD.should_stop() and self.GLOBAL_EP < MAX_GLOBAL_EP:
            # s = self.env.reset()
            # ep_r = 0
            # total_step = 1
            reset()
            while total_step < MAX_TOTAL_STEP:
                try:
                    s = self.env.get_state()
                    a, p = self.AC.choose_action(s)
                    s_, r, done = self.env.step(a)
                    # if done:
                    #     r = 5000

                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        self.AC.update(done, s_, buffer_r, buffer_s, buffer_a)
                        buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        self._record_global_reward_and_print(
                            self.GLOBAL_RUNNING_R, ep_r, self.GLOBAL_EP,
                            total_step)
                        self.GLOBAL_EP += 1
                        reset()

                    # s = s_
                    total_step += 1
                    if self.name == 'W_0':
                        # self.env.render()
                        # time.sleep(0.01)
                        logger.debug([
                            "s ", s, " v ",
                            self.AC.get_v(s), " a ", a, " p ", p
                        ])
                except Exception as e:
                    print(e)

            try:
                print(self.name, " not done,may be donkey!", " total_step:",
                      total_step)
            except Exception as e:
                print(e)

    def work(self):
        buffer_s, buffer_a, buffer_r = [], [], []
        s = self.env.reset()
        ep_r = 0
        total_step = 1

        def reset():
            nonlocal ep_r, total_step
            self.env.reset()
            ep_r = 0
            total_step = 1

        while not COORD.should_stop():
            reset()
            while total_step < MAX_TOTAL_STEP:
                try:
                    s = self.env.get_state()
                    a, p = self.AC.choose_action(s)
                    s_, r, done = self.env.step(a)
                    # st_ac = self.AC.print_AC()
                    # logger.debug(st_ac)

                    ep_r += r
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)

                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        self.AC.update(done, s_, buffer_r, buffer_s, buffer_a)
                        buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        self._record_global_reward_and_print(
                            self.GLOBAL_RUNNING_R, ep_r, self.GLOBAL_EP,
                            total_step)
                        self.GLOBAL_EP += 1
                        reset()

                    # s = s_
                    total_step += 1
                    if self.name == 'W_0':
                        self.env.render()
                        # time.sleep(0.01)
                        logger.debug([
                            "s ", s, " v ",
                            self.AC.get_v(s), " a ", a, " p ", p,
                            ' total_step ', total_step
                        ])
                except Exception as e:
                    print(e)

            try:
                print(self.name, " not done,may be donkey!", " total_step:",
                      total_step)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    GLOBAL_NET_SCOPE = 'Global_Net'
    N_S = MazeEnv.state_space_dim
    N_A = MazeEnv.action_dim

    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        # OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        global_maze = Maze.build(bounds=(30, 30), block_cnt=200)
        # sess, name, N_S, N_A, globalAC, maze=None
        GLOBAL_AC = A3CNet(SESS, GLOBAL_NET_SCOPE, N_S,
                           N_A)  # we only need its params
        workers = []

        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            workers.append(
                Worker(SESS, i_name, N_S, N_A, GLOBAL_AC, global_maze))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    worker_threads = []

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)

    # for worker in workers:
    #     job = lambda: worker.train()
    #     t = threading.Thread(target=job)
    #     t.start()
    #     worker_threads.append(t)
    # COORD.join(worker_threads)

    for i in range(len(workers) - 1):
        worker = workers[i + 1]
        job = lambda: worker.train()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    # COORD.join(worker_threads)
    worker = workers[0]
    worker.work()

    plt.plot(np.arange(len(Worker.GLOBAL_RUNNING_R)), Worker.GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()
