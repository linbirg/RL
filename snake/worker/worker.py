import multiprocessing
import threading
import time
import tensorflow as tf
import numpy as np

import os
import shutil

# import matplotlib.pyplot as plt

# maze
# from env.game_env import SnakeGameEnv

# from ai.A3C import A3CNet

import logger

# OUTPUT_GRAPH = True
# LOG_DIR = './board-log'
# N_WORKERS = multiprocessing.cpu_count()

logger = logger.Logger(show_in_console=False)


class Worker(object):
    GAMMA = 0.9
    MAX_GLOBAL_EP = 50000
    MAX_TOTAL_STEP = 20000
    UPDATE_GLOBAL_ITER = 200

    GLOBAL_RUNNING_R = []
    global_EP = 0

    def __init__(self, COORD, name, net, game=None):
        self.env = game
        self.name = name
        self.AI = net
        # self.saver = tf.train.Saver()
        self.tf_COORD = COORD

    def _record_global_reward_and_print(self, global_runing_rs, ep_r,
                                        global_ep, total_step):
        global_runing_rs.append(ep_r)
        try:
            print(self.name, "Ep:", global_ep,
                  "| Ep_r: %i" % global_runing_rs[-1], "| total step:",
                  total_step)
        except Exception as e:
            print(e)

    def reset(self):
        return self.env.reset()

    def render(self):
        if self.name == 'W_0':
            self.env.render()
            # time.sleep(0.01)

    def do_game(self):
        total_step = 1
        ep_r = 0
        buffer_s, buffer_a, buffer_r = [], [], []

        while total_step < self.MAX_TOTAL_STEP:
            try:
                s = self.env.get_state()
                a, p = self.AI.choose_action(s)
                s_, r, done = self.env.step(a)

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if done or total_step % self.UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net
                    self.AI.update(done, s_, buffer_r, buffer_s, buffer_a)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AI.pull_global()

                if done:
                    self._record_global_reward_and_print(
                        self.GLOBAL_RUNNING_R, ep_r, self.global_EP,
                        total_step)
                    self.global_EP += 1
                    self.reset()

                total_step += 1
                self.render()
                time.sleep(0.01)
                logger.debug([
                    " s_ ", s_, " a ", a, " p ", p, " r ", r, " total_step ",
                    total_step
                ])

            except Exception as e:
                print(e)

    def train(self):
        while not self.tf_COORD.should_stop(
        ) and self.global_EP < self.MAX_GLOBAL_EP:
            self.reset()
            self.do_game()
