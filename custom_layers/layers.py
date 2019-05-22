import numpy as np
import copy
import inspect
import types as python_types
import marshal
import sys
import warnings

from keras import backend as K
import tensorflow as tf
from keras.engine.topology import Layer, InputSpec


class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
        self.built = True

    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            if K.backend() == 'theano':
                weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
            elif K.backend() == 'tensorflow':
                weights = tf.tensordot(self.att_v, K.tanh(y), axes=[[0], [2]])
                # weights = K.tensorflow.python.ops.math_ops.tensordot(self.att_v, K.tanh(y), axes=[0, 2])

        weights = K.softmax(weights)

        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            # out = out.sum(axis=1)
            out = K.sum(out, axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, x, mask):
        return None

    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None
