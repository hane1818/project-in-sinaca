from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, batch_size=10, training=False):
        self._max_grad_norm = 10
        self._vocab_size = 9569
        self._filter_size = 128
        self._rnn_size = 512
        self._batch_size = batch_size
        self._seq_length = 17
        self._is_training = training
        self._candidate_num = 5

        self._input = tf.placeholder(tf.float32, [self._batch_size, self._seq_length, self._candidate_num],
                                     name='input')
        self._labels = tf.placeholder(tf.int32, [self._batch_size, self._seq_length],
                                      name='output')

        with tf.variable_scope('cnn'):
            input = tf.reshape(self._input, [-1, self._candidate_num])
            input_ = []
            for i in tf.unpack(input):
                inp = sum(tf.unpack(tf.to_float(tf.one_hot(tf.to_int64(i), self._vocab_size, 1, 0))))
                # inp = tf.unpack(sum(tf.unpack(tf.to_float(tf.one_hot(tf.to_int64(i), self._vocab_size, 1, 0)))))
                # inp[inp > 0] = 1
                input_.append(inp)
            input_ = tf.reshape(tf.pack(input_), [-1, self._vocab_size, self._seq_length, 1])
            self._conv = []
            W = tf.get_variable('conv1_W', [self._vocab_size, 1, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv1_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv1')
            self._conv.append(tf.nn.relu(conv + b, name='relu1'))

            W = tf.get_variable('conv2_W', [self._vocab_size, 2, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv2_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv2')
            self._conv.append(tf.nn.relu(conv + b, name='relu2'))

            W = tf.get_variable('conv3_W', [self._vocab_size, 3, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv3_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv3')
            self._conv.append(tf.nn.relu(conv + b, name='relu3'))

            W = tf.get_variable('conv4_W', [self._vocab_size, 4, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv4_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv4')
            self._conv.append(tf.nn.relu(conv + b, name='relu4'))

            W = tf.get_variable('conv5_W', [self._vocab_size, 5, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv5_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv5')
            self._conv.append(tf.nn.relu(conv + b, name='relu5'))

            W = tf.get_variable('conv6_W', [self._vocab_size, 6, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv6_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv6')
            self._conv.append(tf.nn.relu(conv + b, name='relu6'))

            W = tf.get_variable('conv7_W', [self._vocab_size, 7, 1, self._filter_size],
                                initializer=tf.random_normal_initializer(),
                                dtype=tf.float32)
            b = tf.get_variable('conv7_b', [self._filter_size], initializer=tf.zeros_initializer, dtype=tf.float32)
            conv = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, self._vocab_size, 1, 1], name='conv7')
            self._conv.append(tf.nn.relu(conv + b, name='relu7'))

        with tf.variable_scope('rnnlm'):
            next_input = tf.concat(3, self._conv)
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

    @property
    def num_stap(self):
        return self._seq_length

    @property
    def output(self):
        return self._output


if __name__ == '__main__':
    batch_size = 1
    m = Model(1)
    x = np.random.uniform(low=0.5, high=13.3, size=[1, 17, 5]).astype(np.float32)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(m._output, feed_dict={m.input: x}).shape)
