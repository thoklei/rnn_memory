from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sys

import ptb_data_generator as dat_gen

from configs import *
import model_functions
if(tf.__version__ == '1.4.0'):
    print("using old data provider")
    import old_ptb_data_provider as d_prov
else:
    print("using new data provider")
    import ptb_data_provider as d_prov

flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("config", "ptb",
    "The configuration to use. See configs.py. Options are: default_ar, default_mnist.")
flags.DEFINE_string("dat_path", None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("sav_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_string("text_path",None,
    "Path to text file to generate word - id mapping.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_string("model", "lstm",
    "Which type of Model to use. Options are: rnn, lstm, irnn, fast_weights, conceptor")
flags.DEFINE_string("mode", "static",
    "Which RNN unrolling mechanism to choose. Options are: static, dynamic")
    
FLAGS = flags.FLAGS

def ptb_model_fn(features, labels, mode, params):
    """
    Model Function
    """

    config = params['config']
    #print("features before:", features) # expecting batchsize x sequence_length x 1
    #print("labels before: ", labels)
    #features = tf.Print(features, [features])
    features = tf.reshape(features, [-1, config.sequence_length])
    
    #print("features after:", features) #bsize x 50
    #print("labels after:",labels)
    #inp = tf.unstack(tf.cast(features,tf.float32), axis=1)

    embedding = tf.get_variable(
          "embedding", [config.vocab_size, config.embedding_size], dtype=config.dtype)
    inputs = tf.nn.embedding_lookup(embedding, features)
    #print("embedding:", inputs) #expecting batchsize x sequence_length x embedding_size

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.reshape(labels, [-1,config.sequence_length])
        inputs = tf.nn.dropout(inputs, config.keep_prob)

    cell = model_functions.get_rnn_cell(params['model'],config)

    inp = tf.unstack(tf.cast(inputs, config.dtype), axis=1) # should yield list of length sequence_length
    #print("len inp: ", len(inp)) # should be 50
    #print("elem: ",inp[0]) # should be 128 x 10.000
    hidden_states, _ = tf.nn.static_rnn(cell, inp, dtype=config.dtype)
    # hidden state = [batchsize, hidden=config.layer_dim]
    softmax_w = tf.get_variable("softmax_w", [config.layer_dim, config.output_dim])
    softmax_b = tf.get_variable("softmax_b", [config.output_dim])

    logits = []

    for state in hidden_states:
        #logits = tf.layers.dense(state, config.output_dim, activation=None)
        logits.append(tf.matmul(state, softmax_w) + softmax_b)

    logits = tf.transpose(tf.stack(logits),[1,0,2])
    #print("logits: ",logits) # expecting 50 x batchsize x 10.000 => 128 x 50 x 10.000
    #logits += 1e-8 # to prevent NaN loss during training

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(logits,axis=2)
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    # Compute loss.
    loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
        logits,
        labels,
        tf.ones([config.batchsize, config.sequence_length], dtype=config.dtype),
        average_across_timesteps=False,
        average_across_batch=True))
    #loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=tf.argmax(logits,axis=2),
                                   name='acc_op')
    perplexity = tf.exp(loss)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('perplexity', perplexity)
    tf.summary.merge_all()

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = config.optimizer
    if(config.clip_gradients):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, config.clip_value_min, config.clip_value_max), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def create_rev_dict(filename):
    words_to_ids = dat_gen._build_vocab(filename)
    ids_to_words = {v: k for k, v in words_to_ids.items()}
    return ids_to_words

def main(_):

    config = Default_PTB_Config()

    if(FLAGS.use_bfp16):
        config.dtype = tf.bfloat16
    else:
        config.dtype = tf.float32

    ids_to_words = create_rev_dict(FLAGS.text_path)

    classifier = tf.estimator.Estimator(
        model_fn=ptb_model_fn,
        model_dir=FLAGS.sav_path,
        params={
            'model': FLAGS.model,
            'config': config
        })
    
    eval_result = classifier.predict(
        input_fn=lambda:d_prov.test_input_fn(FLAGS.dat_path, config))

    for res in eval_result:
        print([ids_to_words[id] for id in res])
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
