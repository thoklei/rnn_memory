from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import data_provider

from autoconceptor import Autoconceptor
from irnn_cell import IRNNCell
from fast_weight_cell import FastWeightCell
from configs import Default_AR_Config, Default_MNIST_Config

flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("config", None,
    "The configuration to use. See configs.py. Options are: default_ar, default_mnist.")
flags.DEFINE_string("data_path", None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_string("model", "fast_weights",
    "Which type of Model to use. Options are: rnn, lstm, irnn, fast_weights, conceptor")
flags.DEFINE_string("task", "mnist",
    "Which task to solve. Options are: mnist, associative_retrieval")

FLAGS = flags.FLAGS

def get_config():
    config = None
    if FLAGS.config == "default_mnist":
        config = Default_MNIST_Config()
    elif FLAGS.config == "default_ar":
        config = Default_AR_Config()
    else:
        raise ValueError("Config not understood. Options are: default_ar, default_mnist.")
    return config


def model_fn(features, labels, mode, params):
    """Model Function"""

    config = params['config']
    #net = tf.feature_column.input_layer(features, params['feature_columns'])

    inp = tf.unstack(tf.cast(features,tf.float32), axis=1)

    if(params['model'] == 'rnn'):
        cell = tf.contrib.rnn.BasicRNNCell(config.layer_dim)
    elif(params['model'] == 'lstm'):
        cell = tf.contrib.rnn.BasicLSTMCell(config.layer_dim)
    elif(params['model'] == 'irnn'):
        cell = IRNNCell(config.layer_dim)
    elif(params['model'] == 'fast_weights'):
        cell = FastWeightCell(config.layer_dim,config.fw_lambda,config.fw_eta, activation=tf.nn.tanh)
    elif(params['model'] == 'autoconceptor'):
        cell = Autoconceptor(config.layer_dim, config.c_alpha, config.c_lambda, config.batchsize)   
    else:
        raise ValueError("Cell type not understood.")

    outputs, _ = tf.nn.static_rnn(cell, inp, dtype=tf.float32)

    logits = tf.layers.dense(outputs[-1], config.output_dim, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(_):

    config = get_config()

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.save_path,
        params={
            'model': FLAGS.model,
            'config':config
        })

    # Train the Model.
    classifier.train(input_fn=lambda:data_provider.train_input_fn(FLAGS.data_path, FLAGS.task, config))

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:data_provider.eval_input_fn(FLAGS.data_path, FLAGS.task, config))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
