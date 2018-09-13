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
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear

#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear

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
                 sequence_length=9,
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
        super(DynamicFastWeightCell, self).__init__(
            num_units, activation, reuse)
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

    def _norm(self, inp, scope="layer_norm"):
        """

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
        """ Run one step of a __BLANK__Cell

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`
            state: A DynStateTuple
        """
        # resetting list of hidden states if a new sequence starts
        if(len(self.hidden_states) == self.sequence_length):
            self.hidden_states = []

        with tf.variable_scope("slow_weights"):
            linear = _linear([inputs, h], self._num_units, True)
        h_s = self._activation(linear)

        # inner loop to update hs to hs+1
        for _ in range(self.num_inner_loops):

            state_sum = tf.zeros([self.batch_size, self._num_units])

            t = len(self.hidden_states)
            for tau, old_hidden in enumerate(self.hidden_states):
                scal_prod = tf.reduce_sum(tf.multiply(tf.matmul(old_hidden, tf.transpose(
                    h_s)), tf.diag(np.ones([self.batch_size], dtype=np.float32))), 1)
                state_sum += tf.multiply(self._lam**(t-tau-1) * old_hidden,
                                         tf.reshape(scal_prod, [self.batch_size, 1]))

            h_A = self._eta * tf.reshape(state_sum, [-1, self._num_units])

            h_pre = linear + h_A

            if(self._layer_norm):
                h_pre = self._norm(h_pre)

            h_s = self._activation(h_pre)

        self.hidden_states.append(h_s)

        return h_s, h_s
