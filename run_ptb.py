from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sys
import numpy as np

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
flags.DEFINE_string("data_path", None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_string("model", "fast_weights",
    "Which type of Model to use. Options are: rnn, lstm, irnn, fast_weights, conceptor")
flags.DEFINE_string("mode", "static",
    "Which RNN unrolling mechanism to choose. Options are: static, dynamic")
    
FLAGS = flags.FLAGS


def get_config():
    config = None
    if FLAGS.config == "mnist_784":
        config = MNIST_784_Config()
    elif FLAGS.config == "default_ar":
        config = Default_AR_Config()
    elif FLAGS.config == "mnist_28":
        config = MNIST_28_Config()
    elif FLAGS.config == "addition":
        config = Default_Addition_Config()
    elif FLAGS.config == "ptb":
        config = Default_PTB_Config()
    else:
        raise ValueError("Config not understood. Options are: default_ar, mnist_784, mnist_28.")
    return config


def ptb_model_fn(features, labels, mode, params):
    """
    Model Function
    """
    global final_state
    global dropout
    global old_hs_1
    global old_hs_2

    config = params['config']

    dropout = tf.placeholder(dtype=tf.float32)

    features = tf.reshape(features, [-1, config.sequence_length])

    old_hs_1 = tf.placeholder(dtype=tf.float32, shape=[config.batchsize,config.layer_dim])
    old_hs_2 = tf.placeholder(dtype=tf.float32, shape=[config.batchsize,config.layer_dim])

    embedding = tf.get_variable(
          "embedding", [config.vocab_size, config.embedding_size], 
          dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, features)

    inputs = tf.nn.dropout(inputs, dropout)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.reshape(labels, [-1,config.sequence_length])[:,1:]
        inputs = inputs[:,:-1]

    #cell = model_functions.get_rnn_cell(params['model'],config)

    cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(config.layer_dim),output_keep_prob=dropout) for _ in range(2)])

    inp = tf.unstack(tf.cast(inputs, tf.float32), axis=1) # should yield list of length sequence_length-1
    #print("len inp: ", len(inp)) # should be sequence_length-1
    #print("elem: ",inp[0]) # should be batchsize x embedding_size
    hidden_states, final_state = tf.nn.static_rnn(cell, inp, 
                                    initial_state=(
                                        (tf.nn.rnn_cell.LSTMStateTuple(c=old_hs_1,h=old_hs_1),
                                        tf.nn.rnn_cell.LSTMStateTuple(c=old_hs_2,h=old_hs_2))), dtype=tf.float32)
    
    print("final state: ",final_state)
    # hidden state = [batchsize, hidden=config.layer_dim]
    #print(hidden_states)
    softmax_w = tf.get_variable("softmax_w", [config.layer_dim, config.vocab_size])
    softmax_b = tf.get_variable("softmax_b", [config.vocab_size])

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

    #print("logits right before: ",logits)
    #print("labels right before: ",labels)
    # Compute loss.
    loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(
        logits=logits,
        targets=labels,
        weights=tf.ones([config.batchsize, config.sequence_length-1], dtype=tf.float32),
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

    optimizer = config.optimizer#tf.train.GradientDescentOptimizer(tf.train.exponential_decay(1.0, tf.train.get_global_step(),
                #                           6*1500, 0.8, staircase=True))
    if(config.clip_gradients):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_value(grad, config.clip_value_min, config.clip_value_max), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=tf.train.get_global_step())
    else:
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)



def main(_):

    config = get_config()

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    with open(os.path.join(FLAGS.save_path,"config.txt"), "w") as text_file:
        print(config, file=text_file)

    classifier = tf.estimator.Estimator(
        model_fn=ptb_model_fn,
        model_dir=FLAGS.save_path,
        params={
            'model': FLAGS.model,
            'config': config
        })


    class FeedHook(tf.train.SessionRunHook):
        def before_run(self, run_context):
            return tf.train.SessionRunArgs(fetches=final_state,
                feed_dict={old_hs_1:hidden_1,old_hs_2:hidden_2})

        def after_run(self, run_context, run_values):
            global hidden_1
            global hidden_2
            hidden_1_state, hidden_2_state = run_values.results
            hidden_1 = hidden_1_state.c
            hidden_2 = hidden_2_state.c

    class DropoutHook(tf.train.SessionRunHook):

        def __init__(self,dropout):
            super(DropoutHook, self).__init__()
            self.dropout = dropout

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(fetches=None,
                feed_dict={dropout:self.dropout})

    feed_hook = FeedHook()
    dropout_train_hook = DropoutHook(config.keep_prob)
    dropout_eval_hook = DropoutHook(1.0)

    save_500_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=FLAGS.save_path,
        save_steps=500)

    for epoch in range(config.num_epochs):

        hidden_1 = np.zeros([config.batchsize, config.layer_dim])
        hidden_2 = np.zeros([config.batchsize, config.layer_dim])

        # Train the Model.
        classifier.train(
            input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, config),
            hooks = [feed_hook, dropout_train_hook, save_500_hook],
            steps=1500) 

        hidden_1 = np.zeros([config.batchsize, config.layer_dim])
        hidden_2 = np.zeros([config.batchsize, config.layer_dim])

        #Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:d_prov.validation_input_fn(FLAGS.data_path, config),
            name="validation",
            hooks = [feed_hook, dropout_eval_hook],
            steps=80)

        print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

    hidden_1 = np.zeros([config.batchsize, config.layer_dim])
    hidden_2 = np.zeros([config.batchsize, config.layer_dim])

    eval_result = classifier.evaluate(
        input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, config),
        name="test",
        hooks = [feed_hook, dropout_eval_hook],
        steps=170)
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
