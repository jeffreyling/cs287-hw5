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
    # if s not in common_words_list:
        # return 'UNK'

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

def word_to_feats(word, global_id, id_to_pos, common_words_list, features_to_idx=None, window=None):
    new_feats = []
    if args.cap > 0:
        if word.isupper():
            new_feats.append('CAP:UPPER')
        elif word[0].isupper():
            new_feats.append('CAP:FIRST')
        elif any(let.isupper() for let in word):
            new_feats.append('CAP:HAS')

    word = clean_str(word, common_words_list)
    if args.suffix > 0:
        for i in range(1, args.suffix+1):
            if len(word) >= i:
                new_feats.append('SUFF:' + word[-1*i:])

        for k,l in enumerate(window):
            for j,w in enumerate(l):
                for i in range(1, args.suffix+1):
                    if len(w) >= i and w != '<s>' and w != '</s>':
                        new_feats.append(('SUFF:%d:%d:' % (k,j)) + w[-1*i:])

    if args.prefix > 0:
        for i in range(1, args.prefix+1):
            if len(word) >= i:
                new_feats.append('PREF:' + word[:i])

        for k,l in enumerate(window):
            for j,w in enumerate(l):
                for i in range(1, args.prefix+1):
                    if len(w) >= i and w != '<s>' and w != '</s>':
                        new_feats.append(('PREF:%d:%d:' % (k,j)) + w[:i])
    if args.pos > 0:
        new_feats.append('POS:' + id_to_pos[global_id])
        if str(int(global_id) + 1) in id_to_pos:
            new_feats.append('POS:1:' + id_to_pos[str(int(global_id)+1)])
        if str(int(global_id) - 1) in id_to_pos:
            new_feats.append('POS:-1:' + id_to_pos[str(int(global_id)-1)])
    if args.all_substr > 0:
        for i in range(len(word)):
            for j in range(i+1, len(word)):
                new_feats.append('SUBSTR:' + word[i:j])

    if features_to_idx:
        return [features_to_idx[feat] for feat in new_feats], word
    else:
        return new_feats, word

def get_words(filename):
    words = []
    words.append(['<s>'])
    sent = 0
    with open(filename, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                words[sent].append('</s>')
                words.append(['<s>'])
                sent += 1
                continue

            word = line[2]
            words[sent].append(word)

    return words

def convert_data(data_name, word_to_idx, features_to_idx, tag_to_id, id_to_pos, common_words_list, dataset, max_sent_len=0, max_feats_len=0, test=False, window_size=5):
    # Construct feature sets for each file. One sentence per row
    idx_features = []
    all_features = []
    lbl = []
    ids = []

    words = get_words(data_name)

    sent = 0
    new_sent = True
    with open(data_name, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) == 0:
                # end padding
                cur_len = len(idx_features[sent])
                idx_features[sent].extend([2] * (max_sent_len - cur_len))
                all_features[sent].extend([[1] * max_feats_len] * (max_sent_len - cur_len))
                ids[sent].extend([0] * (max_sent_len - cur_len))
                if not test:
                    lbl[sent].extend([END_TAG] * (max_sent_len - cur_len))
                sent += 1
                new_sent = True
            else:
                if new_sent:
                    # initial padding
                    idx_features.append([1])
                    all_features.append([[1]*max_feats_len])
                    ids.append([0])
                    if not test:
                        lbl.append([START_TAG])
                    new_sent = False

                global_id = line[0]
                ids[sent].append(global_id)

                # X
                word = line[2]
                i = len(idx_features[sent])
                window = [words[sent][max(i-window_size/2,0) : i], words[sent][i+1: i+window_size/2+1]]
                new_feats, word = word_to_feats(word, global_id, id_to_pos, common_words_list, features_to_idx, window=window)
                new_feats.extend([1]*(max_feats_len - len(new_feats)))
                idx_features[sent].append(word_to_idx[word])
                all_features[sent].append(new_feats)

                if not test:
                    assert len(line) == 4
                    # Y
                    tag = line[3]
                    lbl[sent].append(tag_to_id[tag])

        # end padding for last line
        if sent < len(idx_features):
            cur_len = len(idx_features[sent])
            idx_features[sent].extend([2] * (max_sent_len - cur_len))
            all_features[sent].extend([[1]*max_feats_len] * (max_sent_len - cur_len))
            ids[sent].extend([0] * (max_sent_len - cur_len))
            if not test:
                lbl[sent].extend([END_TAG] * (max_sent_len - cur_len))

    return np.array(idx_features, dtype=np.int32), np.array(all_features, dtype=np.int32), np.array(lbl, dtype=np.int32), np.array(ids, dtype=np.int32)

def get_vocab(file_list, id_to_pos, common_words_list, dataset='', window_size=5):
    # Construct index feature dictionary.
    word_to_idx = {'<s>': 1, '</s>': 2}
    features_to_idx = {'PAD': 1}
    # Start at 3 (1, 2 is start/end of sentence)
    idx = 3
    # Start at 2 (1 is pad)
    feat_idx = 2
    max_sent_len = 0
    max_feats_len = 0
    for filename in file_list:
        if filename:
            words = get_words(filename)
            with open(filename, "r") as f:
                sent_len = 0
                sent = 0
                for line in f:
                    line = line.rstrip()
                    if len(line) == 0:
                        max_sent_len = max(max_sent_len, sent_len)
                        sent_len = 0
                        sent += 1
                        continue
                    sent_len += 1
                    # Add new features
                    global_id = line.split()[0]
                    word = line.split()[2]
                    window = [words[sent][max(sent_len-window_size/2,0) : sent_len], words[sent][sent_len+1: sent_len+window_size/2+1]]
                    new_feats, word = word_to_feats(word, global_id, id_to_pos, common_words_list, window=window)
                    max_feats_len = max(max_feats_len, len(new_feats))
                    if word not in word_to_idx:
                        word_to_idx[word] = idx
                        idx += 1
                    for feat in new_feats:
                        if feat not in features_to_idx:
                            features_to_idx[feat] = feat_idx
                            feat_idx += 1

    max_sent_len += 2 # For start and end padding

    return word_to_idx, features_to_idx, max_sent_len, max_feats_len

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

def shift_window(X, nfeatures, multifeat=False):
    """ Shifts indices so that each position in a window has different features """
    return [x + i*nfeatures for i,x in enumerate(X)]

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
        # Add padding to sentence
        cur_X = [START_WORD]*w + X[i] + [END_WORD]*w
        cur_X_feats = [[1]*num_feats]*w + X_feats[i] + [[1]*num_feats]*w
        cur_Y = [START_TAG]*w + Y[i] + [END_TAG]*w
        for j in xrange(w, len(cur_X) - w):
            if cur_X[j] == cur_X[j-1] and cur_X[j] == END_WORD:
                break
            x = shift_window(cur_X[j-w : j+w+1], V)
            xf = cur_X_feats[j] # single no window
            xf = [feat for feat in xf] # flatten feats
            X_window.append(x)
            X_feats_window.append(xf)
            Y_window.append(cur_Y[j])

    X_window = np.array(X_window, dtype=np.int32)
    X_feats_window = np.array(X_feats_window, dtype=np.int32)
    Y_window = np.array(Y_window, dtype=np.int32)

    # Add previous class to features
    new_col = np.hstack(([START_TAG], Y_window[:-1])).reshape(Y_window.shape[0], 1)
    new_col = np.add(new_col, nfeatures)
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
    parser.add_argument('--suffix', type=int, default=0, help="Suffixes up to specified size")
    parser.add_argument('--prefix', type=int, default=0, help="Prefixes up to specified size")
    parser.add_argument('--pos', type=int, default=0, help="POS tags for word and surrounding words")
    parser.add_argument('--all_substr', type=int, default=0, help="All substrings of a word")
    parser.add_argument('--cap', type=int, default=0, help="Capitalization")
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict = FILE_PATHS[dataset]

    # Get tag to id mapping
    print 'Get tag ids...'
    tag_to_id = get_tag_ids(tag_dict)

    # Retrieve common words
    print 'Getting common words...'
    common_words_list = get_common_words(WORD_VECS_PATH)

    # Get global id to POS mapping
    print 'Getting POS tags...'
    id_to_pos = {}
    with open(POS_TAGS_PATH, 'r') as f:
        # Skip header
        f.next()
        for line in f:
            line = line.strip()
            global_id, tag = line.split(',')
            id_to_pos[global_id] = tag

    # Get index features
    print 'Getting vocab...'
    word_to_idx, features_to_idx, max_sent_len, max_feats_len = get_vocab([train, valid, test], id_to_pos, common_words_list, dataset)
    with open('vocab_list.txt', 'w') as f:
        for k,v in word_to_idx.items():
            f.write("%s\t%d\n" % (k,v))

    V = len(word_to_idx)
    print('Vocab size:', V)
    nfeatures = len(features_to_idx)
    print('Num features:', nfeatures)

    C = len(tag_to_id)

    # Convert data
    print 'Processing data...'
    train_input, train_feats_input, train_output, _ = convert_data(train, word_to_idx, features_to_idx, tag_to_id, id_to_pos, common_words_list, dataset, max_sent_len, max_feats_len)
        # feat_lengths.append(45) # POS tags?

    # To windows
    train_input_window, train_feats_input_window, train_output_window = window_format(train_input.tolist(), train_feats_input.tolist(), train_output.tolist(), V, nfeatures)

    if valid:
        valid_input, valid_feats_input, valid_output, _ = convert_data(valid, word_to_idx, features_to_idx, tag_to_id, id_to_pos, common_words_list, dataset, max_sent_len, max_feats_len)
        # To windows
        valid_input_window, valid_feats_input_window, valid_output_window = window_format(valid_input.tolist(), valid_feats_input.tolist(), valid_output.tolist(), V, nfeatures)

    if test:
        test_input, test_feats_input, _, test_ids = convert_data(test, word_to_idx, features_to_idx, tag_to_id, id_to_pos, common_words_list, dataset, max_sent_len, max_feats_len, test=True)


    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
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
