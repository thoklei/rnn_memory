from tensorflow.python.ops import init_ops
import tensorflow as tf
import numpy as np

class IRNNCell(tf.nn.rnn_cell.BasicRNNCell):

    def __init__(self, num_units, activation=tf.nn.relu, reuse=tf.AUTO_REUSE, dtype=tf.float32):
        super(IRNNCell, self).__init__(num_units, activation, reuse)

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable(
            "kernel",
            shape=[input_depth + self._num_units, self._num_units],
            initializer = init_ops.constant_initializer(value=np.concatenate((np.random.normal(loc=0.0, scale=0.001, size=(input_depth,self._num_units)),np.identity(self._num_units)),0),dtype=self.dtype))

        self._bias = self.add_variable(
            "bias",
            shape=[self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True
