"""
This code is used to perform a grid search over lambda and alpha for the autoconceptor
on the associative retrieval task (which should be easy to change, though).

The way my grid searches work is simple: Three loops, one for each set of hyperparameter values
and one for the number of runs (three should probably be enough). For each run, the estimator
writes everything (i.e. checkpoints, summary...) into the model directory. At the end of each run, the checkpoints
are discarded, while the summary files are written to the summary path, so you end up with a folder with a bunch of
tfevents-files. Also, the configs that were used and the averaged results are written into a results.txt-file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import shutil
from configs import *
import model_functions 
import glob

if(tf.__version__ == '1.4.0'):
    print("using old data provider")
    import old_data_provider as d_prov
else:
    print("using new data provider")
    import data_provider as d_prov



flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("config", "mnist_28",
    "The configuration to use. See configs.py. Options are: default_ar, default_mnist.")
flags.DEFINE_string("data_path",  None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_string("model", "autoconceptor",
    "Which type of Model to use. Options are: rnn, lstm, irnn, fast_weights, conceptor")
flags.DEFINE_string("task", "mnist_28",
    "Which task to solve. Options are: mnist_28, mnist_784, associative_retrieval")
flags.DEFINE_string("mode", "static",
    "Which RNN unrolling mechanism to choose. Options are: static, dynamic")
flags.DEFINE_string("summary_path", None,
"Where to store the summaries of the run (not deleted afterwards)")

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
    config.c_layer_norm = False

    #config.num_epochs = 3 # change this for mnist
    num_runs = 3
    train_steps = 5000
    c_lambdas = [0.001, 0.01, 0.05, 0.1, 0.15, 0.3]
    c_alphas = [20]

    for lam in c_lambdas:
        config.c_lambda = lam

        for alpha in c_alphas:
            config.c_alpha = alpha

            res_list = []

            for run in range(num_runs):
                
                print("Starting run {} of {} for lambda {} and alpha {}".format(run+1, num_runs, lam, alpha))

                model_dir = os.path.join(FLAGS.save_path,"{}_{}_{}".format(run,alpha,lam))
                
                classifier = tf.estimator.Estimator(
                    model_fn=get_model_fn(FLAGS.task, FLAGS.mode),
                    model_dir=model_dir,
                    params={
                        'model': FLAGS.model,
                        'config': config
                    })

                summary_dir = os.path.join(FLAGS.summary_path,"{}_{}".format(lam,alpha),"run_{}".format(run))
             
                for _ in range(10):
                    # Train the Model.
                    classifier.train(
                        input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, FLAGS.task, config)) #500*128 = 64000 = number of training samples
    
                eval_result = classifier.evaluate(
                    input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, FLAGS.task, config),
                    name="test",
                    steps=100
                )
                print("Evaluation complete")
                res_list.append(eval_result['accuracy'])

                event_file = glob.glob(os.path.join(model_dir,"events.out.tfevents*"))
                if not os.path.exists(summary_dir):
                   os.makedirs(summary_dir)
                print(event_file)
                shutil.copy(event_file[0], os.path.join(summary_dir,"events.out.tfevents"))

                shutil.rmtree(model_dir, ignore_errors=True)

            accuracy = np.mean(res_list)

            if not os.path.exists(FLAGS.save_path):
                os.makedirs(FLAGS.save_path)

            with open(os.path.join(FLAGS.save_path,"results.txt"), "a") as text_file:
                conf_info = "\n===== run details =====\n"+str(config)+"\n\nyielded test set accuracy: {}\n=============\n".format(accuracy)
                text_file.write(conf_info)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
