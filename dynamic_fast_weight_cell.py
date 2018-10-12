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

from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_initializer: starting value to initialize the bias
        (default is all zeros).
        kernel_initializer: starting value to initialize the weight.
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if(shape.ndims != 2):
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                         "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
            dtype=dtype,
            initializer=kernel_initializer)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            if bias_initializer is None:
                bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
            biases = vs.get_variable(
                _BIAS_VARIABLE_NAME, [output_size],
                dtype=dtype,
                initializer=bias_initializer)
        return nn_ops.bias_add(res, biases)


class DynamicFastWeightCell(tf.nn.rnn_cell.BasicRNNCell):
    """ 
    ONLY WORKS FOR static_rnn DO NOT USE WITH mode=dynamic

    A FastWeight Cell following Ba et al (2016)
    This cell calculates the weight matrix dynamically as the weighted
    scalar product over old hidden states to save memory.
    """

    def __init__(self, num_units, lam, eta, batch_size,
                 sequence_length,
                 layer_norm=False,
                 norm_gain=1,
                 norm_shift=1,
                 activation=tf.nn.relu,
                 num_inner_loops=1,
                 reuse=tf.AUTO_REUSE,
                 scal_prod_weight=100,
                 dtype=tf.float32):
        """ 
        Initialize parameters for a FastWeightCell

        num_units       = int, Number of units in the recurrent network
        lam             = float value, decay rate of dynamic fast weight matrix
        eta             = float value, update rate of dynamic fast weight matrix
        layer_norm      = bool, switches layer_norm operation, Default: `False`
        norm_gain       = (Required if layer_norm=True) float value, gain/var of layer norm
        norm_shift      = (Required if layer_norm=True) float value, shift/mean of layer norm
        activation      = (optional) specify the activation function, Default: `ReLU`
        batch_size      = size of the training batches, needed to allocate memory properly
        num_inner_loops = the number of inner loops to transform hs to hs+1 (only 1 works properly)
        sequence_length = the length of input sequences, required to allocate memory
        reuse           = whether to reuse variables in existing scope. 

        """
        super(DynamicFastWeightCell, self).__init__(num_units, activation, reuse, dtype)
        self._num_units = num_units
        self._lam = lam
        self._eta = eta
        self.batch_size = batch_size
        self.num_inner_loops = num_inner_loops
        self.sequence_length = sequence_length

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift
        self._activation = activation

        self.hidden_states = []
        self.scal_prod_weight = scal_prod_weight
        #self.scal_prod_weight = tf.get_variable(name="scal_prod_weight",shape=(),initializer=init_ops.constant_initializer(scal_prod_weight))

    def _norm(self, inp, scope="layer_norm"):
        """ 
        Performs layer normalization on the hidden state.

        inp = the input to be normalized
        
        Returns inp normalized by learned parameters gamma and beta
        """
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope, reuse=tf.AUTO_REUSE):
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def call(self, inputs, h):
        """ 
        Run one step of a DynamicFastWeight-Cell

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`
            state: A DynStateTuple
        """
        # resetting list of hidden states if a new sequence starts
        if(len(self.hidden_states) == self.sequence_length):
            self.hidden_states = []

        linear = _linear([inputs, h], self._num_units, True)
        h_s = self._activation(linear)

        # inner loop to update hs to hs+1
        for _ in range(self.num_inner_loops):

            state_sum = tf.zeros([self.batch_size, self._num_units])

            t = len(self.hidden_states)
            for tau, old_hidden in enumerate(self.hidden_states):
                norm_old_hidden = old_hidden / tf.norm(old_hidden,keepdims=True,axis=1)
                norm_h_s = h_s / tf.norm(h_s,keepdims=True,axis=1)
                #print("norm_oh:",norm_old_hidden)
                #print("norm_hs:",norm_h_s) # should both be 128x50
                scal_prod = tf.reduce_sum(tf.multiply(tf.matmul(norm_old_hidden, tf.transpose(
                    norm_h_s)), tf.diag(np.ones([self.batch_size], dtype=np.float32))), 1)
                #scal_prod = tf.Print(scal_prod, [scal_prod],"scal_prod:")
                state_sum += tf.multiply(self._lam**(t-tau-1) * old_hidden,
                                          tf.reshape(tf.multiply(tf.norm(old_hidden,axis=1)*tf.norm(h_s,axis=1),scal_prod), [self.batch_size, 1]))
                #state_sum = tf.Print(state_sum, [state_sum],message="state sum:")
                #tf.norm(old_hidden,axis=1)*tf.norm(h_s,axis=1)
            h_A = self._eta * tf.reshape(state_sum, [-1, self._num_units])

            h_pre = linear + h_A

            if(self._layer_norm):
                h_pre = self._norm(h_pre)

            h_s = self._activation(h_pre)

        self.hidden_states.append(h_s)#(tf.scalar_mul(self.scal_prod_weight,tf.norm(h_s,keepdims=True)))

        return h_s, h_s
