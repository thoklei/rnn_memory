import collections

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

class FastWeightCell(rnn_cell_impl.RNNCell):
    """ A FastWeight Cell following Ba et al (2016)

    TODO: This should overwrite the zero_state function of RNNCell to be applicable
    to the fast-weight matrix as well. //problem_solved.
    """

    def __init__(self, num_units, lam, eta,
                 layer_norm=False,
                 norm_gain=1,
                 norm_shift=1,
                 weights_initializer=None,
                 activation=tf.nn.relu,
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
        super(FastWeightCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._lam = lam
        self._eta = eta

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

        self._state_size = DynStateTuple([num_units, num_units], num_units)


    @property
    def state_size(self):
        """ TODO

        """
        return self._state_size

    @property
    def output_size(self):
        """ TODO

        """
        return  self._num_units

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

    def call(self, inputs, state):
        """ Run one step of a __BLANK__Cell

        Args:
            inputs: `2-D` tensor with shape `[batch_size x input_size]`
            state: A DynStateTuple
        """
        A, h = state
        # update network
        #initializer = tf.random_normal_initializer(stddev=2/input_shape)
        linear = _linear([inputs, h], self._num_units, True) #rnn_cell_impl._linear # shape [?,50]
        # since A is [BATCH x N x N], i.e. for every batch a different A is used,
        # we need to reshape h to work with that
        h_0 = self._activation(linear)
        h_A = tf.reshape(tf.matmul(tf.reshape(h_0, [-1,1,self._num_units]), A), [-1, self._num_units])
        # h_pre = (linear + math_ops.matmul(h_0, A))
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

    def zero_state(self, batch_size, dtype):
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
