import tensorflow as tf

class DefaultConfig(object):

    fw_activation = staticmethod(tf.nn.tanh)
    c_activation = staticmethod(tf.nn.tanh)

    def __init__(self):
        self.num_epochs = 100 #200 = 100.000 steps 
        self.batchsize = 128
        self.layer_dim = 50
        self.layers = 2

        self.fw_layer_norm = True
        self.fw_lambda = 1.1
        self.fw_eta = 0.4
        self.fw_inner_loops = 1
        
        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.105

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

        self.fw_lambda = 0.9
        self.fw_eta = 0.5

        self.tfrecord_dtype = tf.uint8


class MNIST_28_Config(DefaultConfig):

    def __init__(self):
        super(MNIST_28_Config,self).__init__()
        self.input_length = 28
        self.input_dim = 28
        self.output_dim = 10

        self.fw_eta = 0.5
        self.fw_lambda = 0.8

        self.tfrecord_dtype = tf.uint8


class Default_AR_Config(DefaultConfig):

    def __init__(self):
        super(Default_AR_Config,self).__init__()
        self.input_length = 9
        self.input_dim = 26+10+1
        self.output_dim = 26+10+1

        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.105

        self.tfrecord_dtype = tf.int32
        


class Default_Addition_Config(DefaultConfig):

    def __init__(self):
        super(Default_Addition_Config,self).__init__()
        self.batchsize = 64
        self.input_length = 200
        self.input_dim = 2
        self.output_dim = 1
        self.layer_dim = 25

        self.tfrecord_dtype = tf.float32



class Default_PTB_Config(DefaultConfig):

    def __init__(self):
        super(Default_PTB_Config,self).__init__()
        self.num_epochs = 39
        self.batchsize = 32
        self.sequence_length = 35
        self.input_dim = 1
        self.output_dim = 10000
        self.layer_dim = 650
        self.embedding_size = 500
        self.vocab_size = 10000
        self.keep_prob = 0.2 # for dropout
        self.clip_gradients = False
        self.clip_value_min = -5
        self.clip_value_max = 5
        self.clip_value_norm = 5
        self.learning_rate = 1
        #self.optimizer = tf.train.AdamOptimizer()
        #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer

class FW_PTB_Config(Default_PTB_Config):

    def __init__(self):
        super(FW_PTB_Config,self).__init__()

        self.keep_prob = 0.5
        self.embedding_size = 650
        self.layer_dim = 650
        self.clip_gradients = False

        self.fw_lambda = 0.8
        self.fw_eta = 0.4
        self.fw_layer_norm = False

        self.optimizer = tf.train.AdamOptimizer()

class Auto_PTB_Config(Default_PTB_Config):

    def __init__(self):
        super(Auto_PTB_Config,self).__init__()

        self.keep_prob = 0.45
        self.embedding_size = 650
        self.layer_dim = 650
        self.clip_gradients = False

        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.001

        self.optimizer = tf.train.AdamOptimizer()

    
        


