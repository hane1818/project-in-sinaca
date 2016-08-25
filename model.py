from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, batch_size=10, training=False):
        self._max_grad_norm = 10
        self._vocab_size = 9569
        self._rnn_size = 512
        self._batch_size = batch_size
        self._seq_length = 17
        self._is_training = training

        self._input = tf.placeholder(tf.float32, [self._batch_size, self._seq_length, self._vocab_size],
                                     name='input')
        self._labels = tf.placeholder(tf.int32, [self._batch_size, self._seq_length],
                                      name='output')

        with tf.variable_scope('cnn'):
            # W = tf.Variable(tf.random_normal([self._vocab_size + 1, 3, 1, 32], mean=0, stddev=1), dtype=tf.float32,
            #                 name='conv1_W')
            # b = tf.Variable(tf.zeros([32]), name='conv1_b')
            input_ = tf.reshape(self._input, [-1, self._vocab_size, self._seq_length, 1])

            W = tf.get_variable('conv1_W', [self._vocab_size, 3, 1, 32], initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv1_b', [32], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv1 = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv1')
            self._conv1 = tf.nn.relu(conv1 + b, name='relu1')

            """W = tf.Variable(tf.random_normal([self._vocab_size + 1, 2, 1, 32], mean=0, stddev=1), dtype=tf.float32,
                            name='conv2_W')
            b = tf.Variable(tf.zeros([32]), name='conv2_b')"""

            W = tf.get_variable('conv2_W', [self._vocab_size, 2, 1, 32], initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv2_b', [32], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv2 = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv2')
            self._conv2 = tf.nn.relu(conv2 + b, name='relu2')

        with tf.variable_scope('rnnlm'):
            next_input = tf.concat(3, [self._conv1, self._conv2])
            next_input = tf.reshape(next_input, [self._batch_size, self._seq_length, -1])
            # next_input = tf.reshape(tf.tile(next_input, [1, 5, 1]), [self._batch_size * 5, self._seq_length, -1])
            self._cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_size)
            self._initial_state = self._cell.zero_state(batch_size=self._batch_size, dtype=tf.float32)

            outputs = []
            state = self._initial_state
            for i in range(self._seq_length):
                if i > 0: tf.get_variable_scope().reuse_variables()
                next_, state = self._cell(next_input[:, i, :], state)
                outputs.append(next_)

        output = tf.reshape(tf.concat(1, outputs), [-1, self._rnn_size])

        # W = tf.Variable(tf.truncated_normal([self._rnn_size, self._vocab_size + 1]))
        # b = tf.Variable(tf.zeros([self._vocab_size + 1]))
        W = tf.get_variable('softmax_W', [self._rnn_size, self._vocab_size],
                            initializer=tf.truncated_normal_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable('softmax_b', [self._vocab_size], initializer=tf.zeros_initializer, dtype=tf.float32)
        logits = tf.matmul(output, W) + b
        self._output = tf.nn.softmax(logits)
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._labels, [-1])],
            [tf.ones([self._batch_size * self._seq_length], dtype=tf.float32)])

        self._cost = cost = tf.reduce_sum(loss) / self._batch_size
        self._final_state = state

        if not self._is_training:
            return

        self._lr = tf.Variable(1.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          self._max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    @property
    def train_op(self):
        return self._train_op

    @property
    def input(self):
        return self._input

    @property
    def label(self):
        return self._labels

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    @property
    def cost(self):
        return self._cost

    @property
    def batch_size(self):
        return self._batch_size


if __name__ == '__main__':
    m = Model()
    x = np.random.uniform(low=0.5, high=13.3, size=[1, 9579, 17, 1]).astype(np.float32)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(m._output, feed_dict={m.input: x}).shape)
