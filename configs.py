import tensorflow as tf

class DefaultConfig(object):

    fw_activation = staticmethod(tf.nn.tanh)
    c_activation = staticmethod(tf.nn.tanh)

    def __init__(self):
        self.num_epochs = 200 #200 = 100.000 steps 
        self.batchsize = 128
        self.layer_dim = 50

        self.fw_layer_norm = True
        self.fw_lambda = 1.1
        self.fw_eta = 0.4
        self.fw_inner_loops = 1
        
        self.c_layer_norm = False
        self.c_alpha = 12
        self.c_lambda = 0.065

        self.norm_gain =  1
        self.norm_shift = 1
        self.optimizer = tf.train.AdamOptimizer()
        self.clip_gradients = False

    def __repr__(self):
        return "\n".join([str(key)+": "+str(value) for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)])
    
    

class MNIST_784_Config(DefaultConfig):
    
    def __init__(self):
        super(MNIST_784_Config,self).__init__()
        self.input_length = 784
        self.input_dim = 1
        self.output_dim = 10
        self.batchsize = 64

        self.c_alpha = 40
        self.c_lambda = 0.01


class MNIST_28_Config(DefaultConfig):

    def __init__(self):
        super(MNIST_28_Config,self).__init__()
        self.layer_dim = 100
        self.input_length = 28
        self.input_dim = 28
        self.output_dim = 10


class Default_AR_Config(DefaultConfig):

    def __init__(self):
        super(Default_AR_Config,self).__init__()
        self.input_length = 9
        self.input_dim = 26+10+1
        self.output_dim = 26+10+1


class Default_Addition_Config(DefaultConfig):

    def __init__(self):
        super(Default_Addition_Config,self).__init__()
        self.input_length = 100
        self.input_dim = 2
        self.output_dim = 1


