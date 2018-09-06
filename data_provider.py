import tensorflow as tf
import os

def read_dataset(path, mode, batch_size, repeat, seq_length, seq_width, datatype):
    """
    Reads data from .tfrecords-file, decodes it and returns the dataset as a
    tf.data.TFRecordDataset.

    This probably only works for data that was generated using the export_data-script.

    batch_size      int, the batch_size used during training
    num_epochs      int, the number of training epochs
    seq_length      int, the length of the sequences (eg an mnist-image is 784 long)
    seq_width       int, the width of the sequences (dimensionality of the data)
    """

    def _parse_function(example_proto):
        """
        The function that is used to parse the data in the tfrecords-file.
        Both the MNIST-data as well as the AR-data have the column names
        "raw_label" and "raw_sequence", so there is no need for a second parse
        function here.
        If you want to extend this experiment to other datasets, you might need
        to write a different parse function and make read_dataset accept this as
        an argument.
        """
        features = {"raw_sequence": tf.FixedLenFeature([], tf.string, default_value=""),
                    "raw_label": tf.FixedLenFeature([], tf.string, default_value="")}
        parsed_features = tf.parse_single_example(example_proto, features)

        # NOTE: the reason that this will be an int32 is weird and hidden;
        # retrieve sequence
        seq = tf.decode_raw(parsed_features["raw_sequence"], datatype)
        seq.set_shape(seq_length*seq_width)
        seq = tf.reshape(seq, [seq_length, seq_width])
        seq = tf.cast(seq, tf.float32)

        # retrieve labels [i.e. last and to be predicted element of sequence]
        label = tf.decode_raw(parsed_features["raw_label"], datatype)
        label.set_shape(1)
        label = tf.cast(label, tf.int32)

        return seq, label

    training_path = os.path.join(path, mode+'.tfrecords')
    training_dataset = tf.data.TFRecordDataset(training_path)
    training_dataset = training_dataset.map(_parse_function)
    training_dataset = training_dataset.shuffle(10000)
    training_dataset = training_dataset.batch(batch_size, drop_remainder=True)
    if(repeat):
        training_dataset = training_dataset.repeat()

    return training_dataset 


def input_fn(path, task, config, mode, repeat):
    if(task == "mnist_784"):
        return read_dataset(path, mode, config.batchsize, repeat, seq_length=config.input_length, seq_width=config.input_dim, datatype=tf.uint8)
    elif(task=="associative_retrieval"):
        return read_dataset(path, mode, config.batchsize, repeat, seq_length=config.input_length, seq_width=config.input_dim, datatype=tf.int32)
    elif(task == "mnist_28"):
        return read_dataset(path, mode, config.batchsize, repeat, seq_length=config.input_length, seq_width=config.input_dim, datatype=tf.uint8)    
    elif(task == "addition"):
        return read_dataset(path, mode, config.batchsize, repeat, seq_length=config.input_length, seq_width=config.input_dim, datatype=tf.float32)
    else:
        raise ValueError("Task type not understood.")


def train_input_fn(path, task, config):
    return input_fn(path, task, config, 'train', True)


def validation_input_fn(path, task, config):
    return input_fn(path, task, config, 'validation', False)


def test_input_fn(path, task, config):
    return input_fn(path, task, config, 'test', False)