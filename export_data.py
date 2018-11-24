"""

Writes Data to TFRecords format.

Code adapted from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py

"""

import argparse
import numpy as np
import os
import tensorflow as tf
from data_utils import create_data, create_addition_data
from tensorflow.contrib.learn.python.learn.datasets import mnist

flags = tf.flags
flags.DEFINE_string("dataset", None,
    "Specify which dataset to TFRecord. Options are: mnist, associative-retrieval")
flags.DEFINE_string("path", None,
    "Specifiy where to save the TFRecord dataset.")
flags.DEFINE_integer("length", 10,
    "Specify how many values to add (only for addition dataset)")

FLAGS = flags.FLAGS


def _int64_feature(val):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[val]))

def _bytes_feature(val):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[val]))


def record_ar():
    def _convert_to(data_set, name):
        seq, labels = data_set
        num_examples = len(seq)
        seq_length = seq.shape[1] # length of each sequence to be classified
        seq_width = seq.shape[2] # Embedding width

        filename = os.path.join(FLAGS.path, name + '.tfrecords')
        print('Writing', filename)

        with tf.python_io.TFRecordWriter(filename) as writer:
            for idx in range(num_examples):
                raw_seq = seq[idx].tostring()
                raw_label = labels[idx].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'length': _int64_feature(seq_length),
                    'width': _int64_feature(seq_width),
                    'raw_sequence': _bytes_feature(raw_seq),
                    'raw_label': _bytes_feature(raw_label)
                }))
                writer.write(example.SerializeToString())

    _convert_to(create_data(64000), 'train')
    _convert_to(create_data(32000), 'test')
    _convert_to(create_data(16000), 'validation')

    return None


def record_addition(sequence_length):
    """
    This code supports the addition task too, as presented in the IRNN paper. 
    I did not end up conducting tests with it, though.
    """
    def _convert_to(data_set, name):
        seq, labels = data_set
        num_examples = len(seq)
        seq_length = seq.shape[1] # length of each sequence to be classified
        seq_width = seq.shape[2] # Embedding width

        filename = os.path.join(FLAGS.path, name + '.tfrecords')
        print('Writing', filename)

        with tf.python_io.TFRecordWriter(filename) as writer:
            for idx in range(num_examples):
                raw_seq = seq[idx].tostring()
                raw_label = labels[idx].tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'length': _int64_feature(seq_length),
                    'width': _int64_feature(seq_width),
                    'raw_sequence': _bytes_feature(raw_seq),
                    'raw_label': _bytes_feature(raw_label)
                }))
                writer.write(example.SerializeToString())

    _convert_to(create_addition_data(num_samples=100000, sequence_length=sequence_length), 'train')
    _convert_to(create_addition_data(num_samples=20000, sequence_length=sequence_length), 'test')
    _convert_to(create_addition_data(num_samples=10000, sequence_length=sequence_length), 'validation')

    return None



def record_mnist():
    """
    """

    def _convert_to(data_set, name):
      """Converts a dataset to tfrecords."""
      images = data_set.images
      labels = data_set.labels
      num_examples = data_set.num_examples

      if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))
      rows = images.shape[1]
      cols = images.shape[2]
      depth = images.shape[3]

      filename = os.path.join(FLAGS.path, name + '.tfrecords')
      print('Writing', filename)
      with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(num_examples):
          image_raw = images[index].tostring()
          raw_label = labels[index].tostring()
          example = tf.train.Example(
              features=tf.train.Features(
                  feature={
                      'length': _int64_feature(rows*cols),
                      'raw_label': _bytes_feature(raw_label),
                      'raw_sequence': _bytes_feature(image_raw)
                  }))
          writer.write(example.SerializeToString())


    data_sets = mnist.read_data_sets(FLAGS.path,
                                   dtype=tf.uint8,
                                   reshape=False,
                                   validation_size=5000)

    # Convert to Examples and write the result to TFRecords.
    _convert_to(data_sets.train, 'train')
    _convert_to(data_sets.validation, 'validation')
    _convert_to(data_sets.test, 'test')

    return None



def main(_):
    if not FLAGS.dataset:
        raise ValueError("Please specify the dataset")
    if not FLAGS.path:
        raise ValueError("Please specify the path to the save location you want to use")

    if not os.path.exists(FLAGS.path):
        os.makedirs(FLAGS.path)

    if FLAGS.dataset == 'mnist':
        record_mnist()
    elif FLAGS.dataset == 'associative-retrieval':
        record_ar()
    elif(FLAGS.dataset == 'addition'):
        record_addition(FLAGS.length)
    else:
        raise ValueError("dataset not understood, use 'mnist' or 'associative-retrieval'")


if __name__ == '__main__':
    tf.app.run()
