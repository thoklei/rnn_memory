from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sys
import numpy as np
import ptb_data_generator as dat_gen

from configs import *
import model_functions
from autoconceptor import Autoconceptor, DynStateTuple 
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
    global final_state
    global dropout
    global old_hs_1
    global old_hs_2
    global old_hs_3
    global learning_rate

    config = params['config']

    dropout = tf.placeholder(dtype=config.dtype)

    features = tf.reshape(features, [-1, config.sequence_length])

    old_hs_1 = tf.placeholder(dtype=config.dtype, shape=[config.batchsize, config.layer_dim])
    old_hs_2 = tf.placeholder(dtype=config.dtype, shape=[config.batchsize, config.layer_dim, config.layer_dim])
    old_hs_3 = tf.placeholder(dtype=config.dtype, shape=[config.batchsize, config.layer_dim])

    learning_rate = tf.placeholder(dtype=config.dtype, shape=())

    embedding = tf.get_variable(
          "embedding", [config.vocab_size, config.embedding_size], 
          dtype=config.dtype)
    inputs = tf.nn.embedding_lookup(embedding, features)

    inputs = tf.nn.dropout(inputs, dropout)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.reshape(labels, [-1,config.sequence_length])[:,1:]
    
    cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(config.layer_dim),output_keep_prob=dropout),
             tf.contrib.rnn.DropoutWrapper(Autoconceptor(num_units = config.layer_dim, 
                             alpha = config.c_alpha, 
                             lam = config.c_lambda, 
                             batchsize = config.batchsize, 
                             activation=config.c_activation, 
                             layer_norm=False,
                             dtype=config.dtype),output_keep_prob=dropout)])

    inp = tf.unstack(tf.cast(inputs, config.dtype), axis=1) # should yield list of length sequence_length-1

    hidden_states, final_state = tf.nn.static_rnn(cell, inp, 
                                    initial_state=(tf.nn.rnn_cell.LSTMStateTuple(c=old_hs_1,h=old_hs_1),DynStateTuple(C=old_hs_2, h=old_hs_3)),
                                    dtype=config.dtype)
    
    print("final state: ",final_state)
    # hidden state = [batchsize, hidden=config.layer_dim]
    #print(hidden_states)
    softmax_w = tf.get_variable("softmax_w", [config.layer_dim, config.vocab_size], dtype=config.dtype)
    softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=config.dtype)

    logits = []

    for state in range(len(hidden_states)):
        # ignoring the last hidden state, as this would refer to the first element of the next sequence
        #logits = tf.layers.dense(state, config.output_dim, activation=None)
        logits.append(tf.nn.bias_add(tf.matmul(hidden_states[state], softmax_w),softmax_b))

    logits = tf.transpose(tf.stack(logits),[1,0,2])
    #print("logits: ",logits) # expecting sequence_length x batchsize x vocab_size => batchsize x seq_length x vocab_size
    #logits += 1e-8 # to prevent NaN loss during training

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(logits,axis=2)
        print("predictions:",predictions)# expecting batchsize x sequence_length
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    # seq2seq loss doesn't work with float16
    loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(
        logits=tf.cast(logits[:,:-1],tf.float32),
        targets=labels,
        weights=tf.ones([config.batchsize, config.sequence_length-1], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True))
    #loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=tf.argmax(logits[:,:-1],axis=2),
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

    optimizer = config.optimizer#tf.train.GradientDescentOptimizer(learning_rate)
    if(config.clip_gradients):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_norm(grad, config.clip_value_norm), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)




def create_rev_dict(filename):
    words_to_ids = dat_gen._build_vocab(filename)
    ids_to_words = {v: k for k, v in words_to_ids.items()}
    return ids_to_words, words_to_ids




def main(_):

    config = Default_PTB_Config()

    if(FLAGS.use_bfp16):
        config.dtype = tf.bfloat16
    else:
        config.dtype = tf.float32

    ids_to_words,words_to_ids = create_rev_dict(FLAGS.text_path)

    classifier = tf.estimator.Estimator(
        model_fn=ptb_model_fn,
        model_dir=FLAGS.sav_path,
        params={
            'model': FLAGS.model,
            'config': config
        })

    class FeedHook(tf.train.SessionRunHook):

        def __init__(self,initial_lr):
            super(FeedHook, self).__init__()
            self.current_lr = initial_lr

        def adjust_lr(self,new_val):
            self.current_lr = new_val

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                fetches=final_state,
                feed_dict={
                    old_hs_1:hidden_1,
                    old_hs_2:hidden_2,
                    old_hs_3:hidden_3,
                    learning_rate:self.current_lr})

        def after_run(self, run_context, run_values):
            global hidden_1
            global hidden_2
            global hidden_3
            hidden_1, dyn_state_tuple = run_values.results
            hidden_2 = dyn_state_tuple.C
            hidden_3 = dyn_state_tuple.h

    class DropoutHook(tf.train.SessionRunHook):

        def __init__(self,dropout):
            super(DropoutHook, self).__init__()
            self.dropout = dropout

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                fetches=None,
                feed_dict={
                    dropout:self.dropout})

    feed_hook = FeedHook(config.learning_rate)
    dropout_train_hook = DropoutHook(config.keep_prob)
    dropout_eval_hook = DropoutHook(1.0)

    hidden_1 = np.zeros([config.batchsize, config.layer_dim])
    hidden_2 = np.zeros([config.batchsize, config.layer_dim, config.layer_dim])
    hidden_3 = np.zeros([config.batchsize, config.layer_dim])
    
    words = ['the']
    for _ in range(100):
        eval_result = classifier.predict(
            #input_fn=lambda:d_prov.train_input_fn(FLAGS.dat_path, config),
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x=np.zeros((20,35), dtype=np.int32),#{"x": np.ndarray([words_to_ids[word] for word in words])*},
                shuffle=False
            ),
            hooks=[feed_hook, dropout_eval_hook])

        for res in eval_result:
            print([ids_to_words[id] for id in res])
        words = 
        

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
