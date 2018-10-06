from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import sys
sys.path.append("../")
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

    config = params['config']
    #print("features before:", features) # expecting batchsize x sequence_length x 1
    #print("labels before: ", labels)
    #features = tf.Print(features, [features])
    features = tf.reshape(features, [-1, config.sequence_length])
    labels = tf.reshape(labels, [-1,config.sequence_length])
    #print("features after:", features) #bsize x 50
    #print("labels after:",labels)
    #inp = tf.unstack(tf.cast(features,tf.float32), axis=1)

    embedding = tf.get_variable(
          "embedding", [config.input_dim, config.vocab_size], dtype=tf.float32)
    inputs = tf.nn.embedding_lookup(embedding, features)
    #print("embedding:", inputs) #expecting batchsize x sequence_length x vocab_size

    if mode == tf.estimator.ModeKeys.TRAIN:
        inputs = tf.nn.dropout(inputs, config.keep_prob)

    cell = model_functions.get_rnn_cell(params['model'],config)

    inp = tf.unstack(tf.cast(inputs, tf.float32), axis=1) # should yield list of length sequence_length
    #print("len inp: ", len(inp)) # should be 50
    #print("elem: ",inp[0]) # should be 128 x 10.000
    hidden_states, _ = tf.nn.static_rnn(cell, inp, dtype=tf.float32)
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
        tf.ones([config.batchsize, config.sequence_length], dtype=tf.float32),
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
    for epoch in range(config.num_epochs):
        # Train the Model.
        classifier.train(
            input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, config)) 

        #Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:d_prov.validation_input_fn(FLAGS.data_path, config),
            name="validation")

        print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

    eval_result = classifier.evaluate(
        input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, config),
        name="test"
    )
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
