#!/usr/bin/env python

"""NER Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re

# Your preprocessing, features construction, and word2vec code.

def get_tag_ids(tag_dict):
    # Construct tag to id mapping
    tag_to_id = {}
    with open(tag_dict, 'r') as f:
        for line in f:
            tag, id_num = tuple(line.split())
            tag_to_id[tag] = int(id_num)
    # For test set, _ is an unknown tag
    tag_to_id['<t>'] = 8
    tag_to_id['</t>'] = 9
    return tag_to_id

def convert_data(data_name, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, dataset):
    # Construct index feature sets for each file
    idx_features = []
    suffix_features = []
    prefix_features = []
    lbl = []
    ids = []
    max_lbls = 1
    with open(data_name, "r") as f:
        # initial padding
        idx_features.extend([1])
        suffix_features.extend([1])
        prefix_features.extend([1])
        lbl.append([8])
        ids.extend([0])

        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                # add padding
                idx_features.extend([1,2])
                suffix_features.extend([1,2])
                prefix_features.extend([1,2])
                lbl.append([8])
                lbl.append([9])
                ids.extend([-1,0])
            else:
                line = line.split()
                global_id = line[0]
                word = line[2]
                suffix = word[-2:]
                prefix = word[:2]
                tags = [0]
                if len(line) > 3:
                    tags = line[3:]
                    lbl.append([tag_to_id[tag] for tag in tags])
                    if len(tags) > max_lbls:
                        max_lbls = len(tags)

                idx_features.append(word_to_idx[word])
                suffix_features.append(suffix_to_idx[suffix])
                prefix_features.append(prefix_to_idx[prefix])
                ids.append(global_id)
        # end padding
        idx_features.extend([2])
        suffix_features.extend([2])
        prefix_features.extend([2])
        lbl.append([9])
        ids.extend([0])

    # Normalize lbl length
    for i in range(len(lbl)):
        if len(lbl[i]) < max_lbls:
            lbl[i].extend([0] * (max_lbls - len(lbl[i])))

    return np.array(idx_features, dtype=np.int32), np.array(suffix_features, dtype=np.int32), np.array(prefix_features, dtype=np.int32), np.array(lbl, dtype=np.int32), np.array(ids, dtype=np.int32)

def get_vocab(file_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {}
    suffix_to_idx = {}
    prefix_to_idx = {}
    # Start at 3 (1, 2 is start/end of sentence)
    idx = 3
    suffix_idx = 3
    prefix_idx = 3
    for filename in file_list:
        if filename:
            with codecs.open(filename, "r", encoding="latin-1") as f:
=======
                # add padding word
                features.extend([1] * (window_size/2))
                cap_features.extend([1] * (window_size/2))
                suffix_features.extend([1] * (window_size/2))
                lbl.extend([0] * (window_size/2))
                ids.extend([0] * (window_size/2))
            else:
                print line
                global_id, _, word, tag = tuple(line.split())
                lower_caps = int(word.islower())
                all_caps = int(word.isupper())
                first_letter_cap = int(word[0].isupper())
                has_one_cap = int(any(let.isupper() for let in word))
                cap = 1
                if lower_caps:
                    cap = 2
                elif all_caps:
                    cap = 3
                elif first_letter_cap:
                    cap = 4
                elif has_one_cap:
                    cap = 5
                word = clean_str(word, common_words_list)
                suffix = word[-2:]

                features.append(word_to_idx[word])
                cap_features.append(cap)
                suffix_features.append(suffix_to_idx[suffix])
                lbl.append(tag_to_id[tag])
                ids.append(global_id)
        # end padding
        features.extend([1] * (window_size/2))
        cap_features.extend([1] * (window_size/2))
        suffix_features.extend([1] * (window_size/2))
        lbl.extend([0] * (window_size/2))
        ids.extend([0] * (window_size/2))

    # Convert to windowed features
    for i in range(len(features)):
        # Skip padding
        if features[i] == 1:
            continue
        else:
            # window_idxs = range(i - window_size/2, i + window_size/2 + 1)
            i_low = i - window_size/2
            i_high = i + window_size/2 + 1
            window_features.append(features[i_low:i_high])
            window_cap_features.append(cap_features[i_low:i_high])
            window_suffix_features.append(suffix_features[i_low:i_high])
            window_lbl.append(lbl[i])
            window_ids.append(ids[i])
    return np.array(window_features, dtype=np.int32), np.array(window_cap_features, dtype=np.int32), np.array(window_lbl, dtype=np.int32), np.array(window_ids, dtype=np.int32), np.array(window_suffix_features, dtype=np.int32)

def get_vocab(file_list, common_words_list, dataset=''):
    # Construct index feature dictionary.
    word_to_idx = {}
    suffix_to_idx = {}
    # Start at 2 (1 is padding)
    idx = 2
    suffix_idx = 2
    for filename in file_list:
        if filename:
            with open(filename, "r") as f:
                    word = tuple(line.split())[2]
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

    return word_to_idx, suffix_to_idx, prefix_to_idx

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
                        "data/tags.txt",
                        "data/glove.6B.50d.txt")}
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
    train, valid, test, tag_dict, word_vecs = FILE_PATHS[dataset]

    # Fix this for now
    window_size = 5

    # Retrieve tag to id mapping
    print 'Get tag ids...'
    tag_to_id = get_tag_ids(tag_dict)

    # Retrieve common words
    print 'Getting common words...'
    common_words_list = get_common_words(word_vecs)

    # Get index features
    print 'Getting vocab...'
    word_to_idx, suffix_to_idx = get_vocab([train, valid, test], common_words_list, dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_cap_input, train_output, _, train_suffix_input = convert_data(train, word_to_idx, suffix_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if valid:
        valid_input, valid_cap_input, valid_output, _, valid_suffix_input = convert_data(valid, word_to_idx, suffix_to_idx, tag_to_id, window_size, common_words_list, dataset)

    if test:
        test_input, test_cap_input, _, test_ids, test_suffix_input  = convert_data(test, word_to_idx, suffix_to_idx, tag_to_id, window_size, common_words_list, dataset)

    # +4 for cap features
    # V = len(word_to_idx) + 1 + 4
    V = len(word_to_idx) + 1
    print('Vocab size:', V)

    # -1 for _ tag
    C = len(tag_to_id) - 1

    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(word_vecs, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
    for word, vec in word_vecs.items():
        embed[word_to_idx[word] - 1] = vec

    # Get tag to id mapping
    print 'Get tag ids...'
    tag_to_id = get_tag_ids(tag_dict)

    # Get index features
    print 'Getting vocab...'
    word_to_idx, suffix_to_idx, prefix_to_idx = get_vocab([train, valid, test], dataset)

    # Convert data
    print 'Processing data...'
    train_input, train_suffix_input, train_prefix_input, train_output, _ = convert_data(train, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, dataset)

    if valid:
        valid_input, valid_suffix_input, valid_prefix_input, valid_output, _ = convert_data(valid, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, dataset)

    if test:
        test_input, test_suffix_input, test_prefix_input, _, test_ids = convert_data(test, word_to_idx, suffix_to_idx, prefix_to_idx, tag_to_id, dataset)

    # +2 for padding
    V = len(word_to_idx) + 2
    print('Vocab size:', V)

    # -1 for _ tag
    C = len(tag_to_id) - 1

    # Get word vecs
    print 'Getting word vecs...'
    word_vecs = load_word_vecs(WORD_VECS_PATH, word_to_idx)
    embed = np.random.uniform(-0.25, 0.25, (V, len(word_vecs.values()[0])))
    # zero out padding
    embed[0] = 0
    embed[1] = 0
    for word, vec in word_vecs.items():
        embed[word_to_idx[word] - 2] = vec

    print 'Saving...'
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_suffix_input'] = train_suffix_input
        f['train_prefix_input'] = train_prefix_input
        f['train_output'] = train_output
        if valid:
            f['valid_input'] = valid_input
            f['valid_suffix_input' ] = valid_suffix_input
            f['valid_prefix_input'] = valid_prefix_input
            f['valid_output'] = valid_output
        if test:
            f['test_input'] = test_input
            f['test_ids'] = test_ids
        f['nfeatures'] = np.array([V], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['word_vecs'] = embed

        f['word_vecs'] = embed

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
