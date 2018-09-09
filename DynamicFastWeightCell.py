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
    if shape.ndims != 2:
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
    """ A FastWeight Cell following Ba et al (2016)

    TODO: This should overwrite the zero_state function of RNNCell to be applicable
    to the fast-weight matrix as well. //problem_solved.
    """

    def __init__(self, num_units, lam, eta,
                 layer_norm=False,
                 norm_gain=1,
                 norm_shift=1,
                 weights_initializer=None,
                 activation=None,
                 batch_size=128,
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
        self._num_units = num_units
        self._lam = lam
        self._eta = eta
        self.batch_size = batch_size

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
        self._activation = activation or tf.nn.relu

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
        h_0 = self._activation(linear)
        
        # inner loop
        #h_A = tf.reshape(tf.matmul(tf.reshape(h_0, [-1,1,self._num_units]), A), [-1, self._num_units])
        
        state_sum = tf.zeros([self.batch_size,self._num_units])
        t = len(self.hidden_states)
        for tau, old_hidden in enumerate(reversed(self.hidden_states)):
            #scal_prod = tf.reshape(tf.matmul(tf.transpose(old_hidden),h_0),[1, self._num_units, self._num_units])
            #print(scal_prod)
            #state_sum += tf.matmul(tf.reshape(self._lam**(t-tau-1) * old_hidden,[1,-1,self._num_units]),scal_prod) 
            scal_prod = tf.reduce_sum(tf.multiply(tf.matmul(old_hidden,tf.transpose(h_0)),tf.diag(np.ones([self.batch_size], dtype=np.float32))),1)
            #print(scal_prod) # should be b,1
            state_sum += tf.multiply(self._lam**(t-tau-1) * old_hidden,tf.reshape(scal_prod,[self.batch_size,1]))
            #print(state_sum)

        h_A = self._eta * tf.reshape(state_sum,[-1,self._num_units])

        h_pre = linear + h_A

        h_ln = self._norm(h_pre)
        h = self._activation(h_ln)


        self.hidden_states.append(h)
        #self.counter += 1
        return h, h
