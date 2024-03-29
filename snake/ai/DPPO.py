"""
DPPO
"""

import tensorflow as tf
from tensorflow.contrib.distributions import Normal
import numpy as np

# EP_MAX = 2000
# EP_LEN = 300
# N_WORKER = 4  # parallel workers

# MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
# UPDATE_STEP = 5  # loop update operation n-steps

# MODE = ['easy', 'hard']
# n_model = 1

# env = ArmEnv(mode=MODE[n_model])
# S_DIM = env.state_dim
# A_DIM = env.action_dim
# A_BOUND = env.action_bound[1]


class PPO(object):
    # GAMMA = 0.9  # reward discount factor
    A_LR = 0.0001  # learning rate for actor
    C_LR = 0.0005  # learning rate for critic
    EPSILON = 0.2  # Clipped surrogate objective
    A_BOUND = 2

    def __init__(self, SESS, scope, N_S, N_A, glob_ppo=None):
        self.sess = SESS
        self.N_S = N_S
        self.N_A = N_A
        self.g_ppo = glob_ppo

        self.tfs = tf.placeholder(tf.float32, [None, N_S], 'state')

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # choosing action
        self.update_oldpi_op = [
            oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)
        ]

        self.tfa = tf.placeholder(tf.float32, [None, self.N_A], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv  # surrogate loss

        self.aloss = -tf.reduce_mean(
            tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1. - self.EPSILON, 1. + self.EPSILON) *
                self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self, done, s_, ses, aes, res):

        self.sess.run(self.update_oldpi_op)  # old pi to pi

        steps = len(ses)

        adv = self.sess.run(self.advantage, {
            self.tfs: self._reshape(ses[0]),
            self.tfdc_r: res[0]
        })
        [
            self.sess.run(self.atrain_op, {
                self.tfs: self._reshape(ses[i]),
                self.tfa: aes[i],
                self.tfadv: adv
            }) for i in range(steps)
        ]
        [
            self.sess.run(self.ctrain_op, {
                self.tfs: self._reshape(ses[i]),
                self.tfdc_r: res[i]
            }) for i in range(steps)
        ]

    def _reshape(self, m):
        return m[np.newaxis, :]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs,
                                 200,
                                 tf.nn.relu,
                                 trainable=trainable)
            mu = self.A_BOUND * tf.layers.dense(
                l1, self.N_A, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1,
                                    self.N_A,
                                    tf.nn.softplus,
                                    trainable=trainable)
            norm_dist = Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def _softmax(self, x):
        """Compute the softmax in a numerically stable way."""
        x = x - np.max(x)
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]

        a = self._softmax(a)
        action = np.random.choice(range(
            a.shape[0]), p=a.ravel())  # select action w.r.t the actions prob
        return action, max(a.ravel())

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def pull_global(self):
        pass


# class Worker(object):
#     def __init__(self, wid):
#         self.wid = wid
#         self.env = ArmEnv(mode=MODE[n_model])
#         self.ppo = GLOBAL_PPO

#     def work(self):
#         global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
#         while not COORD.should_stop():
#             s = self.env.reset()
#             ep_r = 0
#             buffer_s, buffer_a, buffer_r = [], [], []
#             for t in range(EP_LEN):
#                 if not ROLLING_EVENT.is_set():                  # while global PPO is updating
#                     ROLLING_EVENT.wait()                        # wait until PPO is updated
#                     buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer
#                 a = self.ppo.choose_action(s)
#                 s_, r, done = self.env.step(a)
#                 buffer_s.append(s)
#                 buffer_a.append(a)
#                 buffer_r.append(r)                    # normalize reward, find to be useful
#                 s = s_
#                 ep_r += r

#                 GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size
#                 if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
#                     v_s_ = self.ppo.get_v(s_)
#                     discounted_r = []                           # compute discounted reward
#                     for r in buffer_r[::-1]:
#                         v_s_ = r + GAMMA * v_s_
#                         discounted_r.append(v_s_)
#                     discounted_r.reverse()

#                     bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
#                     buffer_s, buffer_a, buffer_r = [], [], []
#                     QUEUE.put(np.hstack((bs, ba, br)))
#                     if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
#                         ROLLING_EVENT.clear()       # stop collecting data
#                         UPDATE_EVENT.set()          # globalPPO update

#                     if GLOBAL_EP >= EP_MAX:         # stop training
#                         COORD.request_stop()
#                         break

#             # record reward changes, plot later
#             if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
#             else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
#             GLOBAL_EP += 1
#             print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)

# if __name__ == '__main__':
#     GLOBAL_PPO = PPO()
#     UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
#     UPDATE_EVENT.clear()    # no update now
#     ROLLING_EVENT.set()     # start to roll out
#     workers = [Worker(wid=i) for i in range(N_WORKER)]

#     GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
#     GLOBAL_RUNNING_R = []
#     COORD = tf.train.Coordinator()
#     QUEUE = queue.Queue()
#     threads = []
#     for worker in workers:  # worker threads
#         t = threading.Thread(target=worker.work, args=())
#         t.start()
#         threads.append(t)
#     # add a PPO updating thread
#     threads.append(threading.Thread(target=GLOBAL_PPO.update,))
#     threads[-1].start()
#     COORD.join(threads)

#     # plot reward change and testing
#     plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
#     plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
#     env.set_fps(30)
#     while True:
#         s = env.reset()
#         for t in range(400):
#             env.render()
#             s = env.step(GLOBAL_PPO.choose_action(s))[0]