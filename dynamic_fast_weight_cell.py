import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops.rnn_cell_impl import _concat
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear


class DynamicFastWeightCell(tf.nn.rnn_cell.BasicRNNCell):
    """ 
    A FastWeight Cell following Ba et al (2016)
    """

    def __init__(self, num_units, lam, eta,
                 layer_norm=False,
                 norm_gain=1,
                 norm_shift=1,
                 weights_initializer=None,
                 activation=tf.nn.relu,
                 batch_size=128,
                 num_inner_loops=1,
                 reuse=None):
        """ Initialize parameters for a FastWeightCell

        Args:
            num_units: int, Number of units in the recurrent network
            lam: float value, decay rate of dynamic fast weight matrix
            eta: float value, update rate of dynamic fast weight matrix
            layer_norm: bool, switches layer_norm operation, Default: `False`
            norm_gain: (Required if layer_norm=True) float value, gain/var of layer norm
            norm_shift: (Required if layer_norm=True) float value, shift/mean of layer norm
            activation: (optional) specify the activation function, Default: `ReLU`
            reuse: (optional) [cp from rnn_cell_impl] bool, describes whether to reuse
              variables in existing scope. If not `True`, and the existing scope already
              has the given variables, error is raised.

        """
        super(DynamicFastWeightCell, self).__init__(num_units, activation, reuse)
        print(activation)
        self._num_units = num_units
        self._lam = lam
        self._eta = eta
        self.batch_size = batch_size
        self.num_inner_loops = num_inner_loops

        self._layer_norm = layer_norm
        # if self._layer_norm:
        #     if not (norm_gain or norm_shift):
        #         raise NameError("If Layer-norm is used, initial norm_gain and \
        #                          norm_shift have to be defined")
        self._g = norm_gain
        self._b = norm_shift

        # I need the batch size for A matrix in the model!
        # self._batch_s = batch_size

        # self._weights_initializer = weights_initializer or init_ops.RandomUniform(-1, 1) #NOTE: NOT USED
        self._activation = activation 

        self.hidden_states = []

        self.counter = 0

    
    # @property
    # def state_size(self):
    #     """ TODO

    #     """
    #     return self._state_size

    # @property
    # def output_size(self):
    #     """ TODO

    #     """
    #     return  self._num_units


    def _norm(self, inp, scope="layer_norm"):
        """ TODO

        """
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope):
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized


    # def build(self,inputs_shape):
    #     if inputs_shape[1].value is None:
    #         raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
    #                        % inputs_shape)

    #     input_depth = inputs_shape[1].value

    #     self.W_in = self.add_variable(
    #         "w_in",
    #         shape=[input_depth, self._num_units],
    #         initializer = init_ops.random_normal_initializer(),
    #         dtype=tf.float32)

    #     self.W = self.add_variable(
    #         "W",
    #         shape=[self._num_units, self._num_units],
    #         initializer=init_ops.random_normal_initializer(),
    #         dtype=tf.float32)

    #     self.built = True

    def call(self, inputs, state):
        """ Run one step of a __BLANK__Cell

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`
            state: A DynStateTuple
        """
        # if(self.counter == 9):
        #     self.counter = 0
        #     self.hidden_states = []

        h = state
        # update network
        #initializer = tf.random_normal_initializer(stddev=2/input_shape)
        with tf.variable_scope("slow_weights"):
            linear = _linear([inputs, h], self._num_units, True) #rnn_cell_impl._linear # shape [?,50]
        #linear = tf.matmul(inputs,self.W_in) + tf.matmul(state,self.W)
        # since A is [BATCH x N x N], i.e. for every batch a different A is used,
        # we need to reshape h to work with that
        h_s = self._activation(linear)
        
        # inner loop
        #h_A = tf.reshape(tf.matmul(tf.reshape(h_0, [-1,1,self._num_units]), A), [-1, self._num_units])
        for _ in range(self.num_inner_loops):

            state_sum = tf.zeros([self.batch_size,self._num_units])
            t = len(self.hidden_states)
            for tau, old_hidden in enumerate(self.hidden_states):
                #scal_prod = tf.reshape(tf.matmul(tf.transpose(old_hidden),h_0),[1, self._num_units, self._num_units])
                #print(scal_prod)
                #state_sum += tf.matmul(tf.reshape(self._lam**(t-tau-1) * old_hidden,[1,-1,self._num_units]),scal_prod) 
                scal_prod = tf.reduce_sum(tf.multiply(tf.matmul(old_hidden,tf.transpose(h_s)),tf.diag(np.ones([self.batch_size], dtype=np.float32))),1)
                #print(scal_prod) # should be b,1
                state_sum += tf.multiply(self._lam**(t-tau-1) * old_hidden,tf.reshape(scal_prod,[self.batch_size,1]))
                #print(state_sum)

            h_A = self._eta * tf.reshape(state_sum,[-1,self._num_units])        
            
            h_pre = linear + h_A

            h_ln = self._norm(h_pre)
            h_s = self._activation(h_ln)


        self.hidden_states.append(h_s)
        #self.counter += 1
        return h_s, h_s