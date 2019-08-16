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
from env.game_env import SnakeGameEnv, SnakeGameEnvTriAct
from env.game import Game, GameTriAct

from ai.A3C import A3CNet
from worker.worker import Worker

import logger

OUTPUT_GRAPH = True
MODEL_DIR = './board/model.ckpt'
N_WORKERS = multiprocessing.cpu_count()
# MAX_TOTAL_STEP = 20000
# MAX_GLOBAL_EP = 50000

# UPDATE_GLOBAL_ITER = 500

logger = logger.Logger(show_in_console=False)


def exist_cache():
    return OUTPUT_GRAPH and os.path.exists(MODEL_DIR)


if __name__ == "__main__":
    GLOBAL_NET_SCOPE = 'Global_Net'
    N_S = SnakeGameEnv.state_space_dim
    N_A = SnakeGameEnv.action_dim

    SESS = tf.Session()
    COORD = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        global_maze = Game(bounds=(8, 8))

        GLOBAL_AC = A3CNet(SESS, GLOBAL_NET_SCOPE, N_S,
                           N_A)  # we only need its params

        workers = []

        for i in range(N_WORKERS):
            i_name = 'W_%i' % i  # worker name
            ac = A3CNet(SESS, i_name, N_S, N_A, GLOBAL_AC)
            workers.append(Worker(COORD, i_name, ac,
                                  SnakeGameEnv(global_maze)))

    SESS.run(tf.global_variables_initializer())
    worker_threads = []

    # if OUTPUT_GRAPH:
    #     if os.path.exists(LOG_DIR):
    #         shutil.rmtree(LOG_DIR)
    #     tf.summary.FileWriter(LOG_DIR, SESS.graph)

    for i in range(len(workers) - 1):
        worker = workers[i + 1]
        job = lambda: worker.train()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    # COORD.join(worker_threads)
    worker = workers[0]
    if exist_cache():
        worker.load()
    with SESS:
        worker.train()

    # plt.plot(np.arange(len(Worker.GLOBAL_RUNNING_R)), Worker.GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()
