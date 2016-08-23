from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np

vocab_size = 9568
seq_length = 17
batch_size = 2
rnn_size = 512


input_ = tf.placeholder(tf.float32, [batch_size, vocab_size + 1, seq_length, 1], name='input')

#W = tf.Variable(tf.random_normal([vocab_size + 1, 3, 1, 32], mean=0, stddev=1), dtype=tf.float32, name='conv1_W')
W = tf.get_variable('conv1_W', [vocab_size + 1, 3, 1, 32], initializer=tf.random_normal_initializer(), dtype=tf.float32)
b = tf.Variable(tf.zeros([32]), name='conv1_b')
conv1 = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, vocab_size + 1, 1, 1], name='conv1')
conv1 = tf.nn.relu(conv1 + b, name='relu1')

W = tf.Variable(tf.random_normal([vocab_size + 1, 2, 1, 32], mean=0, stddev=1), dtype=tf.float32, name='conv2_W')
b = tf.Variable(tf.zeros([32]), name='conv2_b')
conv2 = tf.nn.conv2d(input_, W, padding='SAME', strides=[1, vocab_size + 1, 1, 1], name='conv2')
conv2 = tf.nn.relu(conv2 + b, name='conv2')

next_input = tf.concat(3, [conv1, conv2])
next_input = tf.reshape(next_input, [batch_size, seq_length, -1])
next_input = tf.reshape(tf.tile(next_input, [1, 5, 1]), [batch_size*5, seq_length, -1])
cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
state = cell.zero_state(batch_size=batch_size*5, dtype=tf.float32)

out = []
for i in range(seq_length):
    if i > 0: tf.get_variable_scope().reuse_variables()
    next_, state = cell(next_input[:, i, :], state)
    out.append(next_)

outs = tf.reshape(tf.concat(1, out), [-1, rnn_size])

W = tf.Variable(tf.truncated_normal([rnn_size, vocab_size+1]))
b = tf.Variable(tf.zeros([vocab_size+1]))
logits = tf.matmul(outs, W) + b
output = tf.nn.softmax(logits)
"""loss = tf.nn.seq2seq.sequence_loss_by_example(
    [logits],
    [tf.reshape(y, [-1])],
    [tf.ones([batch_size * seq_length * 5])])"""

x = np.random.uniform(low=0.5, high=13.3, size=[batch_size, vocab_size + 1, seq_length, 1]).astype(np.float32)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(outs, feed_dict={input_: x}).shape)
    print(sess.run(tf.reshape(tf.argmax(output, 1), [batch_size*5, -1]), feed_dict={input_: x}))
    print(len(sess.run(tf.trainable_variables())))
