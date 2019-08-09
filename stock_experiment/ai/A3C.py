"""
A3C算法
"""
import tensorflow as tf
import numpy as np


class A3CNet(object):
    GLOBAL_NET_SCOPE = 'Global_Net'
    ENTROPY_BETA = 0.1
    LR_A = 0.001  # learning rate for actor
    LR_C = 0.001  # learning rate for critic
    GAMMA = 0.98

    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

    def __init__(self, SESS, scope, N_S, N_A, globalAC=None):
        self.N_S = N_S
        self.N_A = N_A
        self.SESS = SESS

        if scope == A3CNet.GLOBAL_NET_SCOPE:  # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:  # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [
                    None,
                ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1],
                                               'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(
                    scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(
                        tf.log(self.a_prob) * tf.one_hot(
                            self.a_his, N_A, dtype=tf.float32),
                        axis=1,
                        keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(
                        self.a_prob * tf.log(self.a_prob + 1e-5),
                        axis=1,
                        keep_dims=True)  # encourage exploration
                    self.exp_v = A3CNet.ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [
                        l_p.assign(g_p)
                        for l_p, g_p in zip(self.a_params, globalAC.a_params)
                    ]
                    self.pull_c_params_op = [
                        l_p.assign(g_p)
                        for l_p, g_p in zip(self.c_params, globalAC.c_params)
                    ]
                with tf.name_scope('push'):
                    self.update_a_op = A3CNet.OPT_A.apply_gradients(
                        zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = A3CNet.OPT_C.apply_gradients(
                        zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(
                self.s, 800, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a_dp = tf.layers.dropout(l_a, rate=0.4)
            l_a2 = tf.layers.dense(
                l_a_dp,
                400,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la2')
            l_a2_dp = tf.layers.dropout(l_a2)
            l_a3 = tf.layers.dense(
                l_a2_dp,
                200,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la3')
            l_a3_dp = tf.layers.dropout(l_a3)
            l_a4 = tf.layers.dense(
                l_a3_dp,
                100,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la4')
            l_a4_dp = tf.layers.dropout(l_a4)
            l_a5 = tf.layers.dense(
                l_a4_dp,
                50,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='la5')
            l_a5_dp = tf.layers.dropout(l_a5, rate=0.2)
            a_prob = tf.layers.dense(
                l_a5_dp,
                self.N_A,
                tf.nn.softmax,
                kernel_initializer=w_init,
                name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(
                self.s, 800, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            l_c_dp = tf.layers.dropout(l_c, rate=0.5)
            # 多加一层
            l_c2 = tf.layers.dense(
                l_c_dp,
                400,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc2')
            l_c2_dp = tf.layers.dropout(l_c2)
            l_c3 = tf.layers.dense(
                l_c2_dp,
                200,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc3')
            l_c3_dp = tf.layers.dropout(l_c3)
            l_c4 = tf.layers.dense(
                l_c3_dp,
                100,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc4')
            l_c4_dp = tf.layers.dropout(l_c4)
            l_c5 = tf.layers.dense(
                l_c4_dp,
                50,
                tf.nn.relu6,
                kernel_initializer=w_init,
                name='lc5')
            l_c5_dp = tf.layers.dropout(l_c5, rate=0.2)
            v = tf.layers.dense(
                l_c5_dp, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        self.SESS.run([self.update_a_op, self.update_c_op],
                      feed_dict)  # local grads applies to global net

    def update(self, done, s_, buffer_r, buffer_s, buffer_a):
        if done:
            v_s_ = 0  # terminal,应该价值更高？
        else:
            v_s_ = self.get_v(s_)

        buffer_v_target = []
        for r in buffer_r[::-1]:  # reverse buffer r
            v_s_ = r + A3CNet.GAMMA * v_s_
            buffer_v_target.append(v_s_)
        buffer_v_target.reverse()

        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(
            buffer_a), np.vstack(buffer_v_target)
        feed_dict = {
            self.s: buffer_s,
            self.a_his: buffer_a,
            self.v_target: buffer_v_target,
        }
        self.update_global(feed_dict)
        self.pull_global()

    def get_v(self, s):
        v = self.SESS.run(self.v, {self.s: s[np.newaxis, :]})[0, 0]
        return v

    def pull_global(self):  # run by a local
        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = self.SESS.run(
            self.a_prob, feed_dict={
                self.s: s[np.newaxis, :]
            })
        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, max(prob_weights.ravel())
