from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
from random import randint


class DataLoader(object):
    def __init__(self):
        file = json.load(open('data.json'))
        self._ix_to_word = file['ix_to_word']
        self._vocab = file['vocab']
        self._vocab_size = file['vocab_size']
        x, y = file['data']['X'], file['data']['y']
        self._data = self._Data(x, y)
        self._train = self._val = self._test = self._data
        self._data_length = file['data_length']

    def separate_data(self):
        data_X = self._data.X
        data_y = self._data.y
        data_length = self._data_length
        ind = randint(0, data_length)
        if ind < data_length / 2:
            train_X = data_X[ind:ind + data_length // 2]
            train_y = data_y[ind:ind + data_length // 2]
            data_X = data_X[0:ind] + data_X[ind + data_length // 2:data_length]
            data_y = data_y[0:ind] + data_y[ind + data_length // 2:data_length]
        else:
            train_X = data_X[ind - data_length // 2:ind]
            train_y = data_y[ind - data_length // 2:ind]
            data_X = data_X[0:ind - data_length // 2] + data_X[ind:data_length]
            data_y = data_y[0:ind - data_length // 2] + data_y[ind:data_length]

        data_length = len(data_X)
        ind = randint(0, data_length)
        if ind < int(data_length * 0.6):
            val_X = data_X[ind:ind + int(data_length * 0.6)]
            val_y = data_y[ind:ind + int(data_length * 0.6)]
            test_X = data_X[0:ind] + data_X[ind + int(data_length * 0.6):data_length]
            test_y = data_y[0:ind] + data_y[ind + int(data_length * 0.6):data_length]
        else:
            val_X = data_X[ind - int(data_length * 0.6):ind]
            val_y = data_y[ind - int(data_length * 0.6):ind]
            test_X = data_X[0:ind - int(data_length * 0.6)] + data_X[ind:data_length]
            test_y = data_y[0:ind - int(data_length * 0.6)] + data_y[ind:data_length]

        train = self._train = self._Data(train_X, train_y)
        val = self._val = self._Data(val_X, val_y)
        test = self._test = self._Data(test_X, test_y)

        return train, val, test

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def data_length(self):
        return self._data_length

    class _Data(object):
        def __init__(self, datas, labels):
            self._X = datas
            self._y = labels
            self._len = len(datas)
            self._iterator = 0

        def get_batch(self, batch_size):
            i = self._iterator
            if i + batch_size < self._len:
                X, y = self._X[i:i+batch_size], self._y[i:i+batch_size]
                i += batch_size
                self._iterator = i
            else:
                X = self._X[i:i+batch_size] + self._X[0:i + batch_size - self._len]
                y = self._y[i:i+batch_size] + self._y[0:i + batch_size - self._len]
                self._iterator = i + batch_size - self._len

            return X, y

        @property
        def X(self):
            return self._X

        @property
        def y(self):
            return self._y

        def __len__(self):
            return self._len


if __name__ == '__main__':
    d = DataLoader()
    d.separate_data()
    print(len(d.train.get_batch(100)[0][0]))
