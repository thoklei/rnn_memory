
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.contrib.layers.python.layers import layers
np.set_printoptions(threshold=np.inf)

class Autoconceptor(tf.nn.rnn_cell.BasicRNNCell):#tf.nn.rnn_cell.BasicRNNCell):

    def __init__(self, num_units, alpha, lam, batchsize, activation=tf.nn.tanh, layer_norm=False):
        super(Autoconceptor, self).__init__(num_units, activation)
        self.num_units = num_units
        self.c_alpha = alpha
        self.c_lambda = c_lambda
        self.batchsize = batchsize
        self.conceptor_built = False
        self.layer_norm = layer_norm


    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)
        input_dim = inputs_shape[1].value
        with tf.variable_scope('autoconceptor_vars'):

            self.W_in = tf.get_variable(
                "W_in",
                shape=[input_dim, self.num_units],
                initializer=init_ops.random_normal_initializer(),
                dtype=tf.float32)

            self.b_in = tf.get_variable(
                "b_in",
                shape=[self.num_units],
                initializer= init_ops.zeros_initializer(),
                dtype=tf.float32)

            self.W = tf.get_variable(
                "W",
                shape=[self.num_units, self.num_units],
                initializer=init_ops.constant_initializer(0.05 * np.identity(self.num_units)),
                dtype=tf.float32)


            self.gain = tf.get_variable(
                'layer-norm-gain',
                shape=[self.num_units],
                initializer=init_ops.constant_initializer(np.ones([self.num_units])),
                dtype=tf.float32)

            self.bias = tf.get_variable(
                'layer-norm-bias',
                shape=[self.num_units],
                initializer=init_ops.constant_initializer(np.zeros([self.num_units])),
                dtype=tf.float32)


    def build_conceptor(self, batchsize):
        with tf.variable_scope('autoconceptor_vars'):
            self.C = tf.zeros([batchsize, self.num_units, self.num_units],
                dtype=tf.float32, name='weights-rnn-dynamics')

        self.conceptor_built = True

    
    def _norm(self, inp, scope="layer_norm"):
        """ TODO

        """
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(1)
        beta_init = init_ops.constant_initializer(1)
        with tf.variable_scope(scope):
            tf.get_variable("gamma", shape=shape, initializer=gamma_init)
            tf.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized


    def call(self, inputs, state):
        """
        batch x input_length x input_dim
        """

        if(not self.conceptor_built):
            self.build_conceptor(self.batchsize)

        # #with tf.variable_scope('layer_norm'):
        # mu = tf.reduce_mean(state, reduction_indices=0)
        # sigma = tf.sqrt(tf.reduce_mean(tf.square(state - mu),
        #     reduction_indices=0))
        # state = tf.div(self.gain * (state - mu), sigma) + self.bias

        state = self._activation(
            (inputs @ self.W_in + self.b_in) + (state @ self.W) 
        )
        #state = tf.Print(state, [state])
        if(self.layer_norm):
            state = self._norm(state)
        
        state = tf.reshape(state, [-1, 1, self.num_units])

        # THIS DOWN HERE NEEDS TO BE WORKED OUT!
        # ALSO: try original (1/L) conceptor ->
        # TRY: Adding Wh + Ch
        # ...
        aperture_fact = self.c_alpha ** (-2)

        # Std.Version
        self.C = self.C + self.c_lambda * ( tf.transpose((state - state @ self.C), [0,2,1]) @ state \
            - aperture_fact * self.C )

        state = state @ self.C

        #state = tf.Print(state, [state])

        # with tf.variable_scope('layer_norm'):
        #     mu = tf.reduce_mean(h, reduction_indices=0)
        #     sigma = tf.sqrt(tf.reduce_mean(tf.square(h - mu),
        #         reduction_indices=0))
        #     h = tf.div(gain * (h - mu), sigma) + bias
        #
        # # non-linearity
        # h = tf.nn.relu(h)


        # Reshapes necessary for std. matrix multiplication, where one matrix
        # for all elements in batch vs. fast-weights matrix -> different for every
        # element!
        state = tf.reshape(state, [-1, self.num_units])


        return state, state
