from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from dataloader import DataLoader
from model import Model


def run_epoch(session, m, data, eval_op, verbose=False):
    epoch_size = (len(data) // m.batch_size)
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step in range(epoch_size):
        x, y = data.get_batch(m.batch_size)
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input: x,
                                      m.label: y,
                                      m.initial_state: np.array(state)})
        costs += cost
        iters += 1

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main():
    batch_size = 20
    max_epoch = 5000
    data = DataLoader()
    train_data = data.train
    # train_data, valid_data, test_data = data.separate_data()
    # X, y = train.get_batch(batch_size=100)
    # input = [[[0] * data.vocab_size for j in range(len(X[i]))] for i in range(len(X))]

    print("Constructing models......")

    with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope('model', reuse=None):
            m = Model(training=True, batch_size=batch_size)

        """with tf.variable_scope('model', reuse=True):
            mvalid = Model(training=False, batch_size=batch_size)
            mtest = Model(training=False, batch_size=batch_size)"""

        print("Complete!")
        print("Initialize models......")
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=max_epoch)
        latest_chk = tf.train.latest_checkpoint('chk')

        if latest_chk:
            saver.restore(sess, latest_chk)
        else:
            saver.save(sess, 'chk/model-init')

        print("Start training......")

        for i in range(max_epoch):
            # print("Epoch: %d Learning rate: %.3f" % (i + 1, sess.run(m.lr)))
            train_perplexity = run_epoch(sess, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            # valid_perplexity = run_epoch(sess, mvalid, valid_data, tf.no_op())
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            saver.save(sess, 'chk/model', global_step=i + 1)

        print("Complete Training!")
        # print("Start testing......")
        # test_perplexity = run_epoch(sess, mtest, test_data, tf.no_op())
        # print("Test Perplexity: %.3f" % test_perplexity)

        """state = sess.run(m.initial_state)
        cost, state, _ = sess.run([m.cost, m.final_state, m.train_op],
                                  {
                                      m.input: X,
                                      m.label: y,
                                      m.initial_state: state
                                  })

    print(cost)"""


if __name__ == '__main__':
    main()
