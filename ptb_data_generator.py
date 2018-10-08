import os
import collections
import tensorflow as tf
import numpy as np

flags = tf.flags # cmd line FLAG manager for tensorflow

flags.DEFINE_string("data_path", None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_integer("length", 30,
    "The sequence length do be used, defaults to 30")

FLAGS = flags.FLAGS

def write_ptb_to_tfrecords(data_path, save_path, sequence_length):
    train_data, valid_data, test_data, _ = ptb_raw_data(data_path=data_path) #lists of ints
    
    def _convert_to(data_set, name, sequence_length):

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        filename = os.path.join(save_path, name + '.tfrecords')
        print('Writing', filename)

        with tf.python_io.TFRecordWriter(filename) as writer:
            for chunk in chunks(data_set, sequence_length):
                byte_chunk = np.asarray(chunk).tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'raw_sequence': tf.train.Feature(bytes_list = tf.train.BytesList(value=[byte_chunk]))
                }))
                writer.write(example.SerializeToString())
                

    _convert_to(train_data, 'train', sequence_length)
    _convert_to(test_data, 'test', sequence_length)
    _convert_to(valid_data, 'validation', sequence_length)

def ptb_raw_data(data_path=None):
    """Load PTB raw data from data directory "data_path".

    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.

    The PTB dataset comes from Tomas Mikolov's webpage:

    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

    Args:
        data_path: string path to the directory where simple-examples.tgz has
        been extracted.

    Returns:
        tuple (train_data, valid_data, test_data, vocabulary)
        where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
            return f.read().replace("\n", "<eos>").split()


def main():
    write_ptb_to_tfrecords(FLAGS.data_path, FLAGS.save_path, FLAGS.length)

if __name__ == 'main':
    main()
