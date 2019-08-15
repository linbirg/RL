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

from ai.DPPO import PPO
from worker.worker import Worker

import logger

# OUTPUT_GRAPH = True
# LOG_DIR = './board-log'
N_WORKERS = multiprocessing.cpu_count()
# MAX_TOTAL_STEP = 20000
# MAX_GLOBAL_EP = 50000

# UPDATE_GLOBAL_ITER = 500

logger = logger.Logger(show_in_console=False)

if __name__ == "__main__":
    GLOBAL_NET_SCOPE = 'Global_Net'
    N_S = SnakeGameEnvTriAct.state_space_dim
    N_A = SnakeGameEnvTriAct.action_dim

    SESS = tf.Session()
    COORD = tf.train.Coordinator()
    global_maze = GameTriAct(bounds=(12, 12))

    # COORD, SESS, scope, N_S, N_A, glob_ppo=None
    GLOBAL_AC = PPO(SESS, GLOBAL_NET_SCOPE, N_S,
                    N_A)  # we only need its params

    worker = Worker(COORD, '0_worker', GLOBAL_AC,
                    SnakeGameEnvTriAct(global_maze))
    worker.train()
