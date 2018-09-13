import tensorflow as tf

class DefaultConfig(object):
    num_epochs = 200 # should be 100.000 steps 
    batchsize = 128
    layer_dim = 50
    #learning_rate = 1e-4
    #max_grad_norm = 5.0
    fw_lambda = 0.9
    fw_eta = 0.5
    fw_inner_loops = 1
    c_alpha = 40
    c_lambda = 0.01
    layer_norm = True
    #init_scale = 0.1
    norm_gain =  1
    norm_shift = 1
    optimizer = tf.train.AdamOptimizer()
    activation = staticmethod(tf.nn.tanh)

    clip_gradients = False

    def __repr__(self):
        """
        Don't give me that look, if I have to use Python at least let me do this
        """
        return "\n".join([str(key)+": "+str(value) for key, value in DefaultConfig.__dict__.items() if not key.startswith('__') and not callable(key)])
    
    

class MNIST_784_Config(DefaultConfig):
    input_length = 784
    input_dim = 1
    output_dim = 10
    batchsize = 64


class MNIST_28_Config(DefaultConfig):
    layer_dim = 100
    input_length = 28
    input_dim = 28
    output_dim = 10


class Default_AR_Config(DefaultConfig):
    input_length = 9
    input_dim = 26+10+1
    output_dim = 26+10+1


class Default_Addition_Config(DefaultConfig):
    input_length = 100
    input_dim = 2
    output_dim = 1


