
import os
import tensorflow as tf
import sys
import numpy as np
import collections

from configs import *
import model_functions
from fast_weight_cell import FastWeightCell

if(tf.__version__ == '1.4.0'):
    print("using old data provider")
    import old_ptb_data_provider as d_prov
else:
    print("using new data provider")
    import ptb_data_provider as d_prov

flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("config", "default_ptb",
    "The configuration to use. See configs.py. Options are: default_ar, default_mnist.")
flags.DEFINE_string("data_path", None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_bool("train", True,
    "Whether to train the model or not. If not, text is generated from checkpoint.")
    
FLAGS = flags.FLAGS


def get_config():
    config = None
    if FLAGS.config == "fw_ptb":
        config = FW_PTB_Config()
    elif FLAGS.config == "default_ptb":
        config = Default_PTB_Config()
    else:
        raise ValueError("Config not understood. Options are: default_ar, mnist_784, mnist_28.")

    if(FLAGS.use_bfp16):
        config.dtype = tf.float16
    else:
        config.dtype = tf.float32
    return config


def ptb_model_fn(features, labels, mode, params):
    """
    Model Function
    """
    global dropout
    #global learning_rate

    config = params['config']

    dropout = tf.placeholder(dtype=config.dtype)
    sequence_length = features['length']
    features = features['sequence']

    features = tf.reshape(features, [-1, config.sequence_length]) # to get batchsize x 35

    #learning_rate = tf.placeholder(dtype=config.dtype, shape=())

    embedding = tf.get_variable(
          "embedding", [config.vocab_size, config.embedding_size], 
          dtype=config.dtype)
    inputs = tf.nn.embedding_lookup(embedding, features) # should be batchsize x 35 x 500

    inputs = tf.nn.dropout(inputs, dropout)

    #labels, sequence_length = labels
    sequence_length = tf.reshape(sequence_length,shape=[-1])

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.reshape(labels, [-1,config.sequence_length])[:,1:]

    cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(config.layer_dim, initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05), dtype=config.dtype),output_keep_prob=dropout) for _ in range(2)])

    inp = tf.unstack(tf.cast(inputs, config.dtype), axis=1) # should yield list of length sequence_length-1
    sequence_length = tf.Print(sequence_length, [sequence_length])
    hidden_states, final_state = tf.nn.static_rnn(cell, inp, sequence_length=sequence_length, dtype=config.dtype)

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
            mode, loss=perplexity, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()#tf.train.GradientDescentOptimizer(learning_rate)
    if(config.clip_gradients):
        gvs = optimizer.compute_gradients(loss)
        capped_gvs = [(tf.clip_by_norm(grad, config.clip_value_norm), var) for grad, var in gvs]
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
            'config': config
        })


    class DropoutHook(tf.train.SessionRunHook):

        def __init__(self,dropout):
            super(DropoutHook, self).__init__()
            self.dropout = dropout

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                fetches=None,
                feed_dict={
                    dropout:self.dropout})

    dropout_train_hook = DropoutHook(config.keep_prob)
    dropout_eval_hook = DropoutHook(1.0)

    checkpoint_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=FLAGS.save_path,
        save_steps=1000)

    if(FLAGS.train):
        for epoch in range(config.num_epochs):

            # Train the Model.
            classifier.train(
                input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, config),
                hooks = [dropout_train_hook, checkpoint_hook],
                steps=1327) 

            #Evaluate the model.
            eval_result = classifier.evaluate(
                input_fn=lambda:d_prov.validation_input_fn(FLAGS.data_path, config),
                name="validation",
                hooks = [dropout_eval_hook],
                steps=105)

            print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

        eval_result = classifier.evaluate(
            input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, config),
            name="test",
            hooks = [dropout_eval_hook],
            steps=117)
        
        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    def _build_vocab():

        def _read_words(filename):
            with tf.gfile.GFile(filename, "r") as f:
                return f.read().replace('\n', "<eos>").split()

        data = _read_words("/Users/thomasklein/Uni/Bachelorarbeit/ptbtext/ptb.train.txt")

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        return word_to_id

    word_to_id = _build_vocab()

    cue = ['the', 'meaning', 'of', 'life', 'is']
    encoded_cue = [word_to_id[word] for word in cue] + [0]*(config.sequence_length - len(cue))


    generated_text = classifier.predict(
        input_fn = lambda:d_prov.test_input_fn(FLAGS.data_path, config),
        hooks=[dropout_eval_hook]
    )
    # tf.estimator.inputs.numpy_input_fn(
    #                 x={"features": np.asarray(encoded_cue), 
    #                    "length": np.asarray(5)},
    #                 shuffle=False
    #             ),

    def create_rev_dict(words_to_ids):
        ids_to_words = {v: k for k, v in words_to_ids.items()}
        return ids_to_words

    ids_to_words = create_rev_dict(word_to_id)
    print(generated_text)
    #print([ids_to_words[id] for id in [ids for ids in generated_text]])
    for ids in generated_text:
        print([ids_to_words[id] for id in ids])
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
