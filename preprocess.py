#!/usr/bin/env python

"""NER Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.



def get_tag_ids(tag_dict):
    # Construct tag to id mapping
    tag_to_id = {}
    with open(tag_dict, 'r') as f:
        for line in f:
            tag, id_num = tuple(line.split())
            tag_to_id[tag] = int(id_num)
    # For test set, _ is an unknown tag
    tag_to_id['_'] = 0
    return tag_to_id

def convert_data(data_name, word_to_idx, tag_to_id, dataset):
    # Construct index feature sets for each file
    bow_features = []
    lbl = []
    ids = []
    max_lbls = 1
    with open(data_name, "r") as f:
        # initial padding
        bow_features.extend([1])
        lbl.append([0])
        ids.extend([0])

        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # add padding
                bow_features.extend([1])
                lbl.append([0])
                ids.extend([0])
            else:
                line = line.split()
                global_id = line[0]
                word = line[2]
                tags = [0]
                if len(line) > 3:
                    tags = line[3:]
                    lbl.append([tag_to_id[tag] for tag in tags])
                    if len(tags) > max_lbls:
                        max_lbls = len(tags)

                bow_features.append(word_to_idx[word])
                ids.append(global_id)
        # end padding
        bow_features.extend([1])
        lbl.append([0])
        ids.extend([0])

    # Normalize lbl length
    for i in range(len(lbl)):
        if len(lbl[i]) < max_lbls:
            lbl[i].extend([0] * (max_lbls - len(lbl[i])))

    return np.array(bow_features, dtype=np.int32), np.array(lbl, dtype=np.int32), np.array(ids, dtype=np.int32)

def get_vocab(file_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {}
    # Start at 2 (1 is padding)
    idx = 2
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
                for line in f:
                    line = line.rstrip()
                    if len(line) == 0:
                        continue
                    word = tuple(line.split())[2]
                    if word not in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1

    return word_to_idx

def load_word_vecs(file_name, vocab):
    # Get word vecs from glove
    word_vecs = {}
    with open(file_name, "r") as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vals = vals[1:]
            if word in vocab:
                word_vecs[word] = vals

    return word_vecs

FILE_PATHS = {"CONLL": ("data/train.num.txt",
                        "data/dev.num.txt",
                        "data/test.num.txt",
                        "data/tags.txt")}
WORD_VECS_PATH = 'data/glove.6B.50d.txt'
args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict = FILE_PATHS[dataset]

    # Get tag to id mapping
    print 'Get tag ids...'
    tag_to_id = get_tag_ids(tag_dict)

    # Get index features
    print 'Getting vocab...'
    word_to_idx = get_vocab([train, valid, test], dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_output, _ = convert_data(train, word_to_idx, tag_to_id, dataset)

    if valid:
        valid_input, valid_output, _ = convert_data(valid, word_to_idx, tag_to_id, dataset)

    if test:
        test_input, _, test_ids = convert_data(test, word_to_idx, tag_to_id, dataset)

    # +1 for padding
    V = len(word_to_idx) + 1
    print('Vocab size:', V)

    # -1 for _ tag
    C = len(tag_to_id) - 1

    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
    # zero out padding
    embed[0] = 0
    for word, vec in word_vecs.items():
        embed[word_to_idx[word] - 1] = vec

    print 'Saving...'
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
            f['test_ids'] = test_ids
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)

        f['word_vecs'] = embed

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
