import collections

import tensorflow as tf
import numpy as np
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
            _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], #87x50 = 50+37x50
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



# following the desing of LSTM state tuples
_DynStateTuple = collections.namedtuple("DynStateTyple", ("A", "h"))

class DynStateTuple(_DynStateTuple):
    """Tuple used by RNN Models with dynamic weight matricies.

    Stores two elements: `(A, h)` in that order
        where A is the dynamic weight matrix
        and   h is the state of the RNN

    adapted from LSTMStateTuple in tensorflow/python/obs/rnn_cell_impl.py
    """

    __slots__ = ()

    @property
    def dtype(self):
        (A, h) = self
        if A.dtype != h.dtype:
            raise TypeError("Matrix and internal state should agree on type: %s vs %s" %
                            (str(A.dtype), str(h.dtype)))
        return A.dtype

def _zero_state_tuple(state_size, batch_size, dtype):
    """Create a zero state DynamicStateTuple: A zero `3-D` tensor with shape
    `[batch_size x net_size x net_size]` and a zero `2-D` tensor with shape
    `[batch_size x net_size]`
    """
    def get_state_shape(s):
        c = _concat(batch_size, s)
        c_static = _concat(batch_size, s, static=True)
        size = array_ops.zeros(c, dtype=dtype)
        size.set_shape(c_static)
        return size
    # Differs from rnn_cell_impl function here, and is specific to this code.
    return DynStateTuple(*[get_state_shape(s) for s in state_size])

class FastWeightCell(tf.nn.rnn_cell.BasicRNNCell):
    """ 
    A FastWeight Cell following Ba et al (2016)

    """

    def __init__(self, num_units, lam, eta,
                 layer_norm=False,
                 norm_gain=1,
                 norm_shift=1,
                 activation=tf.nn.tanh,
                 reuse=None,
                 kernel_initializer=None,
                 dtype=tf.float32):
        """ 
        Initialize parameters for a FastWeightCell

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
            kernel_initializer: how to initialize the weights. Useful for experiments with IRNN-like 
              initialization.

        """
        super(FastWeightCell, self).__init__(num_units=num_units, activation=activation, reuse=tf.AUTO_REUSE)
        # would be better to pass dtype to this call to superclass-constructor, but in earlier versions
        # of TF, BasicRNNCell did not take dtype as an argument.
        self._num_units = num_units
        self._lam = lam
        self._eta = eta
        self.dtype = dtype

        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

        self._activation = activation

        self.kernel_initializer = kernel_initializer

        self._state_size = DynStateTuple([num_units, num_units], num_units)

    # these two properties are required to pass assert_like_rnn_cell test
    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return  self._num_units

    def _norm(self, inp, scope="layer_norm"):
        """ 
        Performs layer normalization on the hidden state.

        inp = the input to be normalized
        
        Returns inp normalized by learned parameters gamma and beta
        """
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope):
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def call(self, inputs, state):
        """ 
        Run one step of a FastWeight Cell

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`
            state: A DynStateTuple
        """
        A, h = state
        # update network
        linear = _linear([inputs, h], self._num_units, True) #rnn_cell_impl._linear # shape [?,50]
        # since A is [BATCH x N x N], i.e. for every batch a different A is used,
        # we need to reshape h to work with that
        h_0 = self._activation(linear)
        h_A = tf.reshape(tf.matmul(tf.reshape(h_0, [-1,1,self._num_units]), A), [-1, self._num_units])
        h_pre = linear + h_A
        if(self._layer_norm):
            h_pre = self._norm(h_pre)
        h = self._activation(h_pre)

        # update matrix
        A = self._matrix_update(A, h)

        return h, DynStateTuple(A, h)

    def _matrix_update(self, A, h):
        """ Updates a second weight matrix according to the
        fast weight update rule described by Ba et. al. (2016)

        Args:
            A: `3-D` tensor with shape `[batch_size x state_size x state_size]`
                -> the fast weight matrix
            h: `2-D` tensor with shape `[batch_size x state_size]`
                -> the last network state

        Returns:
            A `3-D` tensor with shape `[batch_size x state_size x state_size]`, i.e.
            the new fast weight matrix A
        """
        #NOTE: Might be a case where name_scope is more appropriate! (ops.name_scope)
        with ops.name_scope("fast_weight_update"):
            h_reshape = tf.reshape(h, [-1,1,self._num_units])
            A = math_ops.scalar_mul(self._lam, A) + \
                self._eta * math_ops.matmul(array_ops.transpose(h_reshape, [0,2,1]), h_reshape)
        return A

    def zero_state(self, batch_size, dtype=tf.float32):
        """Return zero-filled state tensors, including fast-weight matrix

        Overloads parent method zero_state inherited from rnn_cell_impl.RNNCell
        and forgoes much of the generality included there

        Args:
            batch_szie: int, float or unit Tesnor representing batch size.
            dtype: the data type to use for the state.

        Returns:
            A DynStateTuple with 2 Tensors of type dtype. The first `N-D-D`
            shaped tensor is the fast weight matrix `[batch_size x ]`

        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            state_size = self.state_size
            return _zero_state_tuple(state_size, batch_size, dtype)
