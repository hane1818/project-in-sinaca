from __future__ import unicode_literals

import json
import re


def read_word_graph(filename):
    with open(filename) as fin:
        word_graph = fin.read()
        word_graph = json.loads(word_graph)

    return word_graph


def read_raw_data(filename):
    with open(filename) as fin:
        raw_data = fin.read()
        raw_data = json.loads(raw_data)

    return raw_data


def read_vocab(filename):
    with open(filename) as fin:
        data = json.loads(fin.read())
        data['0'] = '#START'
        data[str(len(data.keys()))] = '#END'

    data = {int(k): v for k, v in data.items()}
    vocab = {v: k for k, v in data.items()}

    return data, vocab, len(data.keys())


def word_to_ix(wordlist, vocab):
    new_list = []

    for word in wordlist:
        if word not in vocab:
            new_list.append(int(vocab['UNK']))
        else:
            new_list.append(int(vocab[word]))

    return new_list


def main():
    word_graph = read_word_graph('wordgraph_final.json')
    raw_data = read_raw_data('coco_raw_val.json')
    ix_to_word, vocab, vocab_size = read_vocab('vocab.json')

    data = {'data': {'X': [], 'y': []}, 'data_length': 0,
            'ix_to_word': ix_to_word, 'vocab': vocab, 'vocab_size': vocab_size}

    for w in word_graph:
        for r in raw_data:
            if w['filename'] == r['file_path']:

                candidates = [[0, 0, 0, 0, 0]] + w['candidates']

                for caption in r['captions']:
                    new_caption = re.split('\W', caption.lower())
                    while '' in new_caption:
                        new_caption.remove('')

                    new_caption = word_to_ix(new_caption, vocab)
                    while len(new_caption) < len(candidates):
                        new_caption.append(vocab_size - 1)

                    data['data']['X'].append(candidates)
                    data['data']['y'].append(new_caption)
                    data['data_length'] += 1

    json.dump(data, open('data.json', 'w'))


if __name__ == '__main__':
    main()
