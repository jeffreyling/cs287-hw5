import argparse
import sys
from nltk import pos_tag

FILE_PATHS = {"CONLL": ("data/train.num.txt",
                        "data/dev.num.txt",
                        "data/test.num.txt",
                        "data/tags.dict")}

OUTPUT_FILE_PATHS = {"CONLL": ("data/train_pos.txt",
                               "data/valid_pos.txt",
                               "data/test_pos.txt")}

def get_words(file):
    words = []
    with open(file, 'r') as f:
        for line in f:
            if len(line) > 1:
                words.append(line.split()[2])
    return words

def write_file(tagged, file, tag_to_id):
    with open(file, 'w') as f:
        for i in range(len(tagged)):
            f.write(str(i+1) + ',' + tag_to_id[tagged[i][1]] + '\n')


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args()

    train, valid, test, tag_dict = FILE_PATHS[args.dataset]
    train_output, valid_output, test_output = OUTPUT_FILE_PATHS[args.dataset]

    # Read words
    print('Read words...')
    train_words = get_words(train)
    valid_words = get_words(valid)
    test_words = get_words(test)

    # Tag words
    print('Tag words...')
    train_tagged = pos_tag(train_words)
    valid_tagged = pos_tag(valid_words)
    test_tagged = pos_tag(test_words)

    # Make tag to id dict
    print('Get tag IDs...')
    tag_to_id = {}
    with open(tag_dict, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            tag_to_id[line[0]] = line[1]
    print(tag_to_id)

    # Write to file
    print('Write results...')
    write_file(train_tagged, train_output, tag_to_id)
    write_file(valid_tagged, valid_output, tag_to_id)
    write_file(test_tagged, test_output, tag_to_id)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
