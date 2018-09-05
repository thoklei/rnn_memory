import tensorflow as tf

class DefaultConfig(object):
    num_epochs = 200 # should be 100.000 steps 
    batchsize = 128
    layer_dim = 50
    learning_rate = 1e-4
    max_grad_norm = 5.0
    fw_lambda = 0.9
    fw_eta = 0.5
    c_alpha = 12
    c_lambda = 0.065
    init_scale = 0.1
    layer_norm = True
    norm_gain =  1
    norm_shift = 1
    optimizer = tf.train.AdamOptimizer()


class MNIST_784_Config(DefaultConfig):
    input_length = 784
    input_dim = 1
    output_dim = 10
    c_alpha = 50
    c_lambda = 0.01


class MNIST_28_Config(DefaultConfig):
    layer_dim = 100
    input_length = 28
    input_dim = 28
    output_dim = 10
    c_alpha = 40 #40
    c_lambda = 0.01


class Default_AR_Config(DefaultConfig):
    input_length = 9
    input_dim = 26+10+1
    output_dim = 26+10+1
    optimizer = tf.train.AdagradOptimizer(learning_rate=1e-4)
