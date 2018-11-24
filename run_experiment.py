"""
This file is the entry point for most of the experiments conducted for the thesis.
Check the readme for detailed instructions on how to start the experiments.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
if(tf.__version__ == '1.4.0'):
    print("using old data provider")
    import old_data_provider as d_prov
else:
    print("using new data provider")
    import data_provider as d_prov
from configs import *
import model_functions 


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
flags.DEFINE_string("task", "mnist_28",
    "Which task to solve. Options are: mnist_28, mnist_784, associative_retrieval")
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

    if(FLAGS.use_bfp16):
        config.dtype = tf.bfloat16
    else:
        config.dtype = tf.float32
    return config


def get_model_fn(task,mode):
    if(task == "addition"):
        return model_functions.scalar_model_fn
    else:
        if(mode == "static"):
            return model_functions.static_classification_model_fn
        elif(mode == "dynamic"):
            return model_functions.dynamic_classification_model_fn


def main(_):

    config = get_config()

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    with open(os.path.join(FLAGS.save_path,"config.txt"), "w") as text_file:
        print(config, file=text_file)

    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(FLAGS.task, FLAGS.mode),
        model_dir=FLAGS.save_path,
        params={
            'model': FLAGS.model,
            'config': config
        })
    for epoch in range(config.num_epochs):
        # Train the Model.
        classifier.train(
            input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, FLAGS.task, config),
            steps=500) #500*128 = 64000 = number of training samples

        #Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:d_prov.validation_input_fn(FLAGS.data_path, FLAGS.task, config),
            steps=100,
            name="validation")

        print('\nValidation set accuracy after epoch {}: {accuracy:0.3f}\n'.format(epoch+1,**eval_result))

    eval_result = classifier.evaluate(
        input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, FLAGS.task, config),
        name="test"
    )
    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
