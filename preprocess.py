#!/usr/bin/env python

"""NER Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re

# Your preprocessing, features construction, and word2vec code.

START_WORD = 1
END_WORD = 2
START_TAG = 8
END_TAG = 9

def clean_str(s, common_words_list):
    s = s.strip().lower()
    if s not in common_words_list:
        return 'UNK'

    return s

def get_common_words(file_name):
    # Get list of common words from glove file (change to trie if this is slow)
    common_words_list = set()
    with open(file_name, "r") as f:
        for line in f:
            word = line.split(' ')[0]
            common_words_list.add(word)
    return common_words_list


def get_tag_ids(tag_dict):
    # Construct tag to id mapping
    tag_to_id = {}
    with open(tag_dict, 'r') as f:
        for line in f:
            tag, id_num = tuple(line.split())
            tag_to_id[tag] = int(id_num)
    # Start, end of sentence tags
    tag_to_id['<t>'] = START_TAG
    tag_to_id['</t>'] = END_TAG
    return tag_to_id

def convert_data(data_name, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, common_words_list, pos_tags, dataset, max_sent_len=0, test=False):
    # Construct feature sets for each file. One sentence per row
    idx_features = []
    suffix_features = []
    prefix_features = []
    pos_features = []
    lbl = []
    ids = []

    # Load POS tags dictionary
    id_to_pos = {}
    with open(pos_tags, 'r') as f:
        # Skip header
        f.next()
        for line in f:
            line = line.strip()
            global_id, tag = line.split(',')
            id_to_pos[global_id] = tag

    sent = 0
    new_sent = True
    with open(data_name, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                # end padding
                cur_len = len(idx_features[sent])
                idx_features[sent].extend([2] * (max_sent_len - cur_len))
                suffix_features[sent].extend([2] * (max_sent_len - cur_len))
                prefix_features[sent].extend([2] * (max_sent_len - cur_len))
                pos_features[sent].extend([0] * (max_sent_len - cur_len))
                ids[sent].extend([0] * (max_sent_len - cur_len))
                if not test:
                    lbl[sent].extend([END_TAG] * (max_sent_len - cur_len))
                sent += 1
                new_sent = True
            else:
                if new_sent:
                    # initial padding
                    idx_features.append([1])
                    suffix_features.append([1])
                    prefix_features.append([1])
                    pos_features.append([0])
                    ids.append([0])
                    if not test:
                        lbl.append([START_TAG])
                    new_sent = False

                global_id = line[0]
                ids[sent].append(global_id)

                # X
                word = clean_str(line[2], common_words_list)
                suffix = word[-2:]
                prefix = word[:2]
                idx_features[sent].append(word_to_idx[word])
                suffix_features[sent].append(suffix_to_idx[suffix])
                prefix_features[sent].append(prefix_to_idx[prefix])
                pos_features[sent].append(id_to_pos[global_id])

                if not test:
                    assert len(line) == 4
                    # Y
                    tag = line[3]
                    lbl[sent].append(tag_to_id[tag])

        # end padding for last line
        if sent < len(idx_features):
            cur_len = len(idx_features[sent])
            idx_features[sent].extend([2] * (max_sent_len - cur_len))
            suffix_features[sent].extend([2] * (max_sent_len - cur_len))
            prefix_features[sent].extend([2] * (max_sent_len - cur_len))
            pos_features[sent].extend([0] * (max_sent_len - cur_len))
            ids[sent].extend([0] * (max_sent_len - cur_len))
            if not test:
                lbl[sent].extend([END_TAG] * (max_sent_len - cur_len))

    return np.array(idx_features, dtype=np.int32), np.array(suffix_features, dtype=np.int32), np.array(prefix_features, dtype=np.int32), np.array(pos_features, dtype=np.int32), np.array(lbl, dtype=np.int32), np.array(ids, dtype=np.int32)

def get_vocab(file_list, common_words_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {'<s>': 1, '</s>': 2}
    suffix_to_idx = {'<s>': 1, '</s>': 2}
    prefix_to_idx = {'<s>': 1, '</s>': 2}
    # Start at 3 (1, 2 is start/end of sentence)
    idx = 3
    suffix_idx = 3
    prefix_idx = 3
    max_sent_len = 0
    for filename in file_list:
        if filename:
            with open(filename, "r") as f:
                sent_len = 0
                for line in f:
                    line = line.rstrip()
                    if len(line) == 0:
                        max_sent_len = max(max_sent_len, sent_len)
                        sent_len = 0
                        continue
                    word = clean_str(line.split()[2], common_words_list)
                    sent_len += 1
                    suffix = word[-2:]
                    prefix = word[:2]
                    if word not in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1
                    if suffix not in suffix_to_idx:
                        suffix_to_idx[suffix] = suffix_idx
                        suffix_idx += 1
                    if prefix not in prefix_to_idx:
                        prefix_to_idx[prefix] = prefix_idx
                        prefix_idx += 1

    max_sent_len += 2 # For start and end padding

    return word_to_idx, suffix_to_idx, prefix_to_idx, max_sent_len

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

def merge_feats(feat_list, feat_lengths):
    nfeatures = 0
    merged_feats = []
    for feats, l in zip(feat_list, feat_lengths):
        shifted_feats = [[f + nfeatures for f in row] for row in feats]
        merged_feats.append(shifted_feats)
        nfeatures += l

    return np.swapaxes(np.swapaxes(np.array(merged_feats), 0,1), 1,2)

def shift_window(X, num_feats, multifeat=False):
    """ Shifts indices so that each position in a window has different features """
    if multifeat:
        return [[el + i*num_feats for el in x] for i,x in enumerate(X)]
    else:
        return [x + i*num_feats for i,x in enumerate(X)]

def window_format(X, X_feats, Y, V, nfeatures):
    """ Transform sentence format X, Y to window format for MEMM """
    N = len(X)
    num_feats = len(X_feats[0][0])
    window_size = 5 # fix for now
    w = window_size / 2

    X_window = []
    X_feats_window = []
    Y_window = []
    for i in xrange(N):
        # Add padding
        cur_X = [START_WORD]*w + X[i] + [END_WORD]*w
        cur_X_feats = [[START_WORD]*num_feats]*w + X_feats[i] + [[END_WORD]*num_feats]*w
        cur_Y = [START_TAG]*w + Y[i] + [END_TAG]*w
        for j in xrange(w, len(cur_X) - w):
            if cur_X[j] == cur_X[j-1] and cur_X[j] == END_WORD:
                break
            x = shift_window(cur_X[j-w : j+w+1], V)
            X_window.append(x)
            xf = shift_window(cur_X_feats[j-w : j+w+1], nfeatures, multifeat=True)
            xf = [el for feats in xf for el in feats] # flatten feats
            X_feats_window.append(xf)
            Y_window.append(cur_Y[j])

    X_window = np.array(X_window, dtype=np.int32)
    X_feats_window = np.array(X_feats_window, dtype=np.int32)
    Y_window = np.array(Y_window, dtype=np.int32)

    # Add previous class to features
    new_col = np.hstack(([START_TAG], Y_window[:-1])).reshape(Y_window.shape[0], 1)
    new_col = np.add(new_col, nfeatures*window_size)
    X_feats_window = np.concatenate((X_feats_window, new_col), axis=1)

    return X_window, X_feats_window, Y_window


FILE_PATHS = {"CONLL": ("data/train.num.txt",
                        "data/dev.num.txt",
                        "data/test.num.txt",
                        "data/tags.txt")}
WORD_VECS_PATH = 'data/glove.6B.50d.txt'
POS_TAGS_PATH = 'CONLL_pred.test'
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

    # Retrieve common words
    print 'Getting common words...'
    common_words_list = get_common_words(WORD_VECS_PATH)

    # Get index features
    print 'Getting vocab...'
    word_to_idx, suffix_to_idx, prefix_to_idx, max_sent_len = get_vocab([train, valid, test], common_words_list, dataset)

    V = len(word_to_idx)
    print('Vocab size:', V)

    C = len(tag_to_id)
    feat_lengths = [len(suffix_to_idx), len(prefix_to_idx)]
    nfeatures = sum(feat_lengths)
    print('Num features:', nfeatures)

    # Convert data
    print 'Processing data...'
    train_input, train_suffix_input, train_prefix_input, train_pos_input, train_output, _ = convert_data(train, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, common_words_list, POS_TAGS_PATH, dataset, max_sent_len)
    train_feats_input = merge_feats([train_suffix_input, train_prefix_input], feat_lengths)

    # To windows
    train_input_window, train_feats_input_window, train_output_window = window_format(train_input.tolist(), train_feats_input.tolist(), train_output.tolist(), V, nfeatures)

    if valid:
        valid_input, valid_suffix_input, valid_prefix_input, valid_pos_input, valid_output, _ = convert_data(valid, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, common_words_list, POS_TAGS_PATH, dataset, max_sent_len)
        valid_feats_input = merge_feats([valid_suffix_input, valid_prefix_input], feat_lengths)

        # To windows
        valid_input_window, valid_feats_input_window, valid_output_window = window_format(valid_input.tolist(), valid_feats_input.tolist(), valid_output.tolist(), V, nfeatures)

    if test:
        test_input, test_suffix_input, test_prefix_input, test_pos_input, _, test_ids = convert_data(test, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, common_words_list, POS_TAGS_PATH, dataset, max_sent_len, test=True)
        test_feats_input = merge_feats([test_suffix_input, test_prefix_input], feat_lengths)


    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
    # zero out padding
    # embed[0] = 0
    # embed[1] = 0
    for word, vec in word_vecs.items():
        embed[word_to_idx[word] - 1] = vec

    print train_input.shape, train_feats_input.shape, train_output.shape, train_input_window.shape, train_feats_input_window.shape
    print train_input_window, train_feats_input_window
    print 'Saving...'
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_features_input'] = train_feats_input
        f['train_output'] = train_output
        f['train_input_window'] = train_input_window
        f['train_feats_input_window'] = train_feats_input_window
        f['train_output_window'] = train_output_window
        if valid:
            f['valid_input'] = valid_input
            f['valid_features_input'] = valid_feats_input
            f['valid_output'] = valid_output
            f['valid_input_window'] = valid_input_window
            f['valid_feats_input_window'] = valid_feats_input_window
            f['valid_output_window'] = valid_output_window
        if test:
            f['test_input'] = test_input
            f['test_features_input'] = test_feats_input
            f['test_ids'] = test_ids
        f['vocab_size'] = np.array([V], dtype=np.int32)
        f['nfeatures'] = np.array([nfeatures], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)

        f['word_vecs'] = embed

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
