from dataloader import DataLoader
from model import Model

import tensorflow as tf


def main():
    data = DataLoader()
    data.separate_data()
    X, y = data.train.get_batch(10)
    #input = [[[0] * data.vocab_size for j in range(len(X[i]))] for i in range(len(X))]
    for i, batch in enumerate(X):
        for t, value in enumerate(batch):
            n_hot = [0] * data.vocab_size
            for v in value:
                n_hot[v] = 1
            X[i][t] = n_hot

    """for i, batch in enumerate(y):
        for t, value in enumerate(batch):
            one_hot = [0] * data.vocab_size
            one_hot[value] = 1
            y[i][t] = one_hot

    print(y)"""
    #

    with tf.Graph().as_default(), tf.Session() as sess:
        m = Model(training=True)
        sess.run(tf.initialize_all_variables())
        state = sess.run(m.initial_state)
        cost, state, _ = sess.run([m.cost, m.final_state, m.train_op],
                                  {
                                      m.input: X,
                                      m.label: y,
                                      m.initial_state: state
                                  })

    print(cost)


if __name__ == '__main__':
    main()
