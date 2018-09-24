from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import numpy as np
import shutil
import glob
from configs import *
import model_functions 


if(tf.__version__ == '1.4.0'):
    print("using old data provider")
    import old_data_provider as d_prov
else:
    print("using new data provider")
    import data_provider as d_prov



flags = tf.flags # cmd line FLAG manager for tensorflow
logging = tf.logging # logging manager for tensorflow

flags.DEFINE_string("config", "default_ar",
    "The configuration to use. See configs.py. Options are: default_ar, default_mnist.")
flags.DEFINE_string("data_path",  None,
    "Where the dataset is stored. Make sure to point to the correct type (MNIST, AR)")
flags.DEFINE_string("save_path", None,
    "Model output directory. This is where event files and checkpoints are stored.")
flags.DEFINE_bool("use_bfp16", False,
    "Train using 16-bit truncated floats instead of 32-bit float")
flags.DEFINE_string("model", "fast_weights",
    "Which type of Model to use. Options are: rnn, lstm, irnn, fast_weights, conceptor")
flags.DEFINE_string("task", "associative_retrieval",
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

    config.num_epochs = 10# change this for mnist
    num_runs = 3

    fw_lambdas = [0.8,0.9,1.0,1.1,1.2]
    fw_etas = [0.4,0.5,0.6,0.7]

    for lam in fw_lambdas:
        config.fw_lambda = lam

        for eta in fw_etas:
            config.fw_eta = eta

            res_list = []

            for run in range(num_runs):

                print("Starting run {} of {} for lambda {} and eta {}".format(run+1, num_runs, lam, eta))

                model_dir = os.path.join(FLAGS.save_path,"{}_{}_{}".format(run,lam, eta))
                
                classifier = tf.estimator.Estimator(
                    model_fn=get_model_fn(FLAGS.task, FLAGS.mode),
                    model_dir=model_dir,
                    params={
                        'model': FLAGS.model,
                        'config': config
                    })

                summary_dir = os.path.join(FLAGS.summary_path,"{}_{}".format(lam,eta),"run_{}".format(run))
             
                for _ in range(config.num_epochs):
                    # Train the Model.
                    classifier.train(
                        input_fn=lambda:d_prov.train_input_fn(FLAGS.data_path, FLAGS.task, config),
                        steps=500) #500*128 = 64000 = number of training samples

                eval_result = classifier.evaluate(
                    input_fn=lambda:d_prov.test_input_fn(FLAGS.data_path, FLAGS.task, config),
                    name="test"
                )
        
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
