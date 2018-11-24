"""
This file contains a bunch of configurations, where a config is just a set of variables.
These are encapsulated in classes to enable inheritance, which makes it possible to easily override
some properties, while leaving the others intact.
So if you are using a Config, make sure that not only the variables defined in the class are set like
you want them, but also the ones it inherites. If you need to change one of those, only change your subclass.
"""
import tensorflow as tf

class DefaultConfig(object):
    """
    Global base class, contains default values
    """

    # activation functions
    fw_activation = staticmethod(tf.nn.tanh)
    c_activation = staticmethod(tf.nn.tanh)

    def __init__(self):
        self.num_epochs = 100 #200 = 100.000 steps, how many training epochs (one run over all examples)
        self.batchsize = 128 # how many examples per batch
        self.layer_dim = 50 # how many units per layer
        self.layers = 2 # how many layers, I don't think this is used anywhere

        self.fw_layer_norm = True # whether to apply layer norm as presented by Ba et al.
        self.fw_lambda = 1.1 # which lambda to use for Fast Weights, see Ba et al.
        self.fw_eta = 0.4 # which eta to use for Fast Weights.
        # lambda = 1.1 and eta = 0.4 works well for associative retrieval but is NOT a good value set
        # for pretty much any other task, because lambda > 1, see thesis for remarks
        self.fw_inner_loops = 1 # how many inner loops to perform in the dynamic calculation of Fast Weights
        # not relevant for standard FW-implementation, which just does 1 step
        self.c_layer_norm = False # whether to layer-norm the conceptor
        self.c_alpha = 20 # alpha value for conceptor, see Jaeger 2016
        self.c_lambda = 0.105 # lambda value for conceptor, see Jaeger 2016

        self.norm_gain =  1 # initial value for norm gain (layer normalization)
        self.norm_shift = 1 # initial value for norm shift (layer normalization)
        self.optimizer = tf.train.AdamOptimizer() # default Optimizer to use. Adam usually just works
        self.clip_gradients = False # whether to clip gradients
        # check if the model function you are using interprets this as clipping by norm or clipping by value

    def __repr__(self):
        """
        This function is called when your print(config). The idea is to print a full representation of all the 
        settings that were used. I print this summary to a text file that I store in the checkpoint directory,
        which might come in handy if you are wondering which exact set of parameters was used for a certain run.
        """
        return "\n".join([str(key)+": "+str(value) for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)])
    

class MNIST_784_Config(DefaultConfig):
    """
    Configuration specifically for the MNIST-task, where every pixel is one input.
    """
    
    def __init__(self):
        super(MNIST_784_Config,self).__init__()
        self.input_length = 784 # total sequence length
        self.input_dim = 1 # dimensionality of a single element of the sequence
        self.output_dim = 10 # dimension of output = number of classes
        self.batchsize = 64

        self.c_alpha = 40
        self.c_lambda = 0.01
        self.c_layer_norm = True

        self.fw_lambda = 0.9
        self.fw_eta = 0.5

        self.tfrecord_dtype = tf.uint8 # in which datatype the data is encoded in the tfrecords file


class MNIST_28_Config(DefaultConfig):
    """
    Config for MNIST row by row
    """

    def __init__(self):
        super(MNIST_28_Config,self).__init__()
        self.input_length = 28
        self.input_dim = 28
        self.output_dim = 10

        self.fw_eta = 0.5
        self.fw_lambda = 0.8

        self.tfrecord_dtype = tf.uint8


class Default_AR_Config(DefaultConfig):
    """
    Config for associative retrieval task
    """

    def __init__(self):
        super(Default_AR_Config,self).__init__()
        self.input_length = 9 # length of the sequence, i.e. three pairs, ??, query
        self.input_dim = 26+10+1 # inputs are encoded as one-hot vectors: alphabet + numbers + ?
        self.output_dim = 26+10+1

        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.105

        self.tfrecord_dtype = tf.int32
        

class Default_Addition_Config(DefaultConfig):
    """
    Config for the addition task as presented in the IRNN paper.
    This is not even mentioned in the thesis because it is a pretty ridiculous task, but
    apparently IRNNs are pretty good at this.
    """

    def __init__(self):
        super(Default_Addition_Config,self).__init__()
        self.batchsize = 64
        self.input_length = 200
        self.input_dim = 2
        self.output_dim = 1
        self.layer_dim = 25

        self.tfrecord_dtype = tf.float32


class Default_PTB_Config(DefaultConfig):
    """
    Default config for the Penn Treebank language modelling task.

    This is the one used for the baseline, but you have to do some more stuff to
    exactly replicate the baseline: use GradientDescentOptimizer with a learning rate
    that is reduced by factor 1.2 every 6 epochs and clipped by norm 5.

    Alternatively, just use Adam without clipping the gradients, should get close.
    """

    def __init__(self):
        super(Default_PTB_Config,self).__init__()
        self.num_epochs = 39
        self.batchsize = 32
        self.input_dim = 1
        self.output_dim = 10000 # vocabulary size: 10,000 words
        self.layer_dim = 650 # like in baseline paper
        self.embedding_size = 650 # I always used embeddings as large as the layers
        # I saw that in a tutorial somewhere, don't know if that is the best way
        self.vocab_size = 10000
        self.keep_prob = 0.5 # for dropout
        self.clip_gradients = True
        self.clip_value_min = -5
        self.clip_value_max = 5
        self.clip_value_norm = 5
        self.learning_rate = 1
        #self.optimizer = tf.train.GradientDescentOptimizer


class FW_PTB_Config(Default_PTB_Config):
    """
    Config for Fast Weights on PTB
    """

    def __init__(self):
        super(FW_PTB_Config,self).__init__()

        self.keep_prob = 0.4
        self.embedding_size = 650
        self.layer_dim = 650
        self.clip_gradients = False

        self.fw_lambda = 0.8
        self.fw_eta = 0.4
        self.fw_layer_norm = True

        self.optimizer = tf.train.AdamOptimizer()


class Auto_PTB_Config(Default_PTB_Config):
    """
    Config for Autoconceptor on PTB
    """

    def __init__(self):
        super(Auto_PTB_Config,self).__init__()

        self.keep_prob = 0.4
        self.embedding_size = 650
        self.layer_dim = 100
        self.clip_gradients = False

        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.001

        self.optimizer = tf.train.AdamOptimizer()


class IRNN_PTB_Config(Default_PTB_Config):
    """
    Config for 4-layer IRNN on PTB
    """

    def __init__(self):
        super(IRNN_PTB_Config,self).__init__()

        self.keep_prob = 0.5
        self.embedding_size = 650
        self.layer_dim = 300
        self.clip_gradients = False

        self.optimizer = tf.train.AdamOptimizer()


class Auto_LSTM_PTB_Config(Default_PTB_Config):
    """
    Config for LSTM-cell + Autoconceptor on PTB
    """

    def __init__(self):
        super(Auto_LSTM_PTB_Config,self).__init__()

        self.keep_prob = 0.4
        self.embedding_size = 650
        self.lstm_layer_dim = 500
        self.auto_layer_dim = 300
        self.layer_dim = self.auto_layer_dim
        self.clip_gradients = False

        self.c_layer_norm = False
        self.c_alpha = 20
        self.c_lambda = 0.001

        self.optimizer = tf.train.AdamOptimizer()
    

class FW_LSTM_PTB_Config(FW_PTB_Config):
    """
    Config for LSTM + Fast Weight cell on PTB
    """

    def __init__(self):
        super(FW_LSTM_PTB_Config,self).__init__()

        self.keep_prob = 0.4
        self.embedding_size = 650
        self.lstm_layer_dim = 500
        self.fw_layer_dim = 300
        self.layer_dim = self.fw_layer_dim
        self.clip_gradients = False

        self.optimizer = tf.train.AdamOptimizer()        


class Multi_FW_PTB_Config(FW_PTB_Config):
    """
    Config for multiple Fast Weight cells on PTB
    (This requires a lot of memory if unrolling RNN statically)
    """

    def __init__(self):
        super(Multi_FW_PTB_Config,self).__init__()

        self.keep_prob = 0.4
        self.embedding_size = 650
        self.layer_dim = 500
        self.clip_gradients = False

        self.optimizer = tf.train.AdamOptimizer()     


