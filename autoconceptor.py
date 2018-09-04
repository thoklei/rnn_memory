
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

class Autoconceptor(tf.nn.rnn_cell.BasicRNNCell):#tf.nn.rnn_cell.BasicRNNCell):

    def __init__(self, num_units, c_alpha, c_lambda, batchsize, activation=tf.nn.tanh, reuse=None):
        super(Autoconceptor, self).__init__(num_units, activation, reuse)
        self.num_units = num_units
        self.c_alpha = c_alpha
        self.c_lambda = c_lambda
        self.batchsize = batchsize
        self.conceptor_built = False


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
                initializer=init_ops.zeros_initializer(),
                dtype=tf.float32)


    def build_conceptor(self, batchsize):
        with tf.variable_scope('autoconceptor_vars'):
            self.C = tf.zeros([batchsize, self.num_units, self.num_units],
                dtype=tf.float32, name='weights-rnn-dynamics')

        self.conceptor_built = True


    def call(self, inputs, state):
        """
        batch x input_length x input_dim
        """

        if(not self.conceptor_built):
            self.build_conceptor(self.batchsize)


        # non-linearity here?
        # h = tf.nn.relu(
        #     (input_tensor[:,t,:] @ W_in + b_in) + (h @ W)
        # )
        state = tf.nn.tanh(
            (inputs @ self.W_in + self.b_in) + (state @ self.W)
        )
        
        # with tf.variable_scope('layer_norm'):
        #     mu = tf.reduce_mean(h, reduction_indices=0)
        #     sigma = tf.sqrt(tf.reduce_mean(tf.square(h - mu),
        #         reduction_indices=0))
        #     h = tf.div(gain * (h - mu), sigma) + bias
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
