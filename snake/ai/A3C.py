"""
A3C算法
"""
import tensorflow as tf
import numpy as np


class A3CNet(object):
    GLOBAL_NET_SCOPE = 'Global_Net'
    ENTROPY_BETA = 0.1
    LR_A = 0.01  # learning rate for actor
    LR_C = 0.005  # learning rate for critic
    GAMMA = 0.9

    # OPT_A = tf.train.AdamOptimizer.RMSPropOptimizer(LR_A, name='RMSPropA')
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
                        tf.log(self.a_prob) *
                        tf.one_hot(self.a_his, N_A, dtype=tf.float32),
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
        drop_rate = 0.16
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            # _, l_a2_dp = self._add_layer('la2', 400, self.s, drop_rate / 4,
            #                              w_init)
            _, l_a3_dp = self._add_layer('la3', 200, self.s, drop_rate / 4,
                                         w_init)
            _, l_a4_dp = self._add_layer('la4', 100, l_a3_dp, drop_rate / 4,
                                         w_init)
            _, l_a5_dp = self._add_layer('la5', 50, l_a4_dp, drop_rate / 8,
                                         w_init)
            _, l_a6_dp = self._add_layer('la6', 20, l_a4_dp, drop_rate / 16,
                                         w_init)

            a_prob = tf.layers.dense(l_a6_dp,
                                     self.N_A,
                                     tf.nn.softmax,
                                     kernel_initializer=w_init,
                                     name='ap')
        with tf.variable_scope('critic'):
            # _, l_c2_dp = self._add_layer('lc2', 400, self.s, drop_rate / 4,
            #                              w_init)
            _, l_c3_dp = self._add_layer('lc3', 200, self.s, drop_rate / 4,
                                         w_init)
            _, l_c4_dp = self._add_layer('lc4', 100, l_c3_dp, drop_rate / 4,
                                         w_init)
            _, l_c5_dp = self._add_layer('lc5', 50, l_c4_dp, drop_rate / 8,
                                         w_init)
            _, l_c6_dp = self._add_layer('lc6', 20, l_c5_dp, drop_rate / 16,
                                         w_init)

            v = tf.layers.dense(l_c6_dp,
                                1,
                                kernel_initializer=w_init,
                                name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def _add_layer(self, name, size, last, drop_rate, w_i, method=tf.nn.relu6):
        ly = tf.layers.dense(last,
                             size,
                             method,
                             kernel_initializer=w_i,
                             name=name)
        ly_dp = tf.layers.dropout(ly, rate=drop_rate)

        return ly, ly_dp

    def update_global(self, feed_dict):  # run by a local
        self.SESS.run([self.update_a_op, self.update_c_op],
                      feed_dict)  # local grads applies to global net

    def update(self, done, s_, buffer_r, buffer_s, buffer_a):
        if done:
            v_s_ = -1
        else:
            v_s_ = self.get_v(s_)
        # v_s_ = self.get_v(s_)

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
        prob_weights = self.SESS.run(self.a_prob,
                                     feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(
            range(prob_weights.shape[1]),
            p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action, max(prob_weights.ravel())
