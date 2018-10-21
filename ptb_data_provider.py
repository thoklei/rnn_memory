import tensorflow as tf
import os


def read_dataset(path, mode, batch_size, repeat):
    """
    Reads data from .tfrecords-file, decodes it and returns the dataset as a
    tf.data.TFRecordDataset.

    batch_size      int, the batch_size used during training
    num_epochs      int, the number of training epochs
    seq_length      int, the length of the sequences (eg an mnist-image is 784 long)
    seq_width       int, the width of the sequences (dimensionality of the data)
    """
    

    def _parse_function(example_proto):
        """
        """
        features = {"raw_sequence": tf.FixedLenFeature([], tf.string, default_value=""),
                    "sentence_length": tf.FixedLenFeature([], tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)

        length = tf.decode_raw(parsed_features["sentence_length"], tf.int64)
        length.set_shape(1)
        length = tf.reshape(length, shape=[1])
        length = tf.cast(length, tf.int32)
        #print(length)

        seq = tf.decode_raw(parsed_features["raw_sequence"], tf.int64)
        seq.set_shape(35)
        seq = tf.reshape(seq, [35])
        seq = tf.cast(seq, tf.int32)

        return {'sequence':seq,'length':length}, seq

    training_path = os.path.join(path, mode+'.tfrecords')
    training_dataset = tf.data.TFRecordDataset(training_path)
    training_dataset = training_dataset.map(_parse_function)
    training_dataset = training_dataset.shuffle(100)
    training_dataset = training_dataset.batch(batch_size, drop_remainder=True)

    if(repeat):
        training_dataset = training_dataset.repeat()

    training_dataset = training_dataset.prefetch(1)
    return training_dataset
    

def input_fn(path, config, mode, repeat):
    return read_dataset(path, mode, config.batchsize, repeat)


def train_input_fn(path, config):
    return input_fn(path, config, 'train', False)


def validation_input_fn(path, config):
    return input_fn(path, config, 'validation', False)


def test_input_fn(path, config):
    return input_fn(path, config, 'test', False)