#!/usr/bin/env python

import numpy as np

import tensorflow as tf
from tensorflow import keras as K


def create_linear(**kwargs):
    return K.layers.Dense(**kwargs)


def arange(start: int, end: int, dtype=tf.float32):
    return tf.range(start, end, dtype=dtype)


class Attention(object):

    def __init__(self, **kwargs: dict):
        super(Attention, self).__init__()

        dropout_prob = kwargs.get('dropout_prob', 0.1)
        is_training = kwargs.get('is_training', True)

        self.dropout = K.layers.Dropout(dropout_prob) \
            if dropout_prob else None
        self.is_training = is_training
        self._attn_score = None

    def __call__(self, query, key, value, mask=None):
        assert query.shape == key.shape, \
            'Query and key dont have same dimensions'

        scores = tf.matmul(query, key, transpose_b=True) /\
            tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))

        if mask is not None:
            scores += (mask * 1E-9)

        attn_score = tf.nn.softmax(scores, axis=-1)
        attn_score = self.dropout(attn_score, self.is_training) \
            if self.dropout else attn_score

        self._attn_score = attn_score
        return tf.matmul(attn_score, value)

    @property
    def attn_score(self):
        return self._attn_score


class MultiHeadAttention(K.layers.Layer):

    def __init__(self, h: int = 8, d_model: int = 512, **kwargs):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, 'Invalid'

        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h

        activation = tf.nn.relu
        dense_args = dict(units=self.d_k, activation=activation)
        self._linears = [[create_linear(**dense_args)
                          for _ in range(h)] for _ in range(3)]

        self._output = create_linear(units=d_model,
                                     activation=activation)
        self._attention = Attention(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs: list, mask=None):
        """
        inputs: query, key, value
        """
        assert len(inputs) == 3, 'Invalid input size'

        outputs = [layers[i](inp)
                   for i, inp in enumerate(inputs) for layers in zip(
                       self._linears[0], self._linears[1],
                       self._linears[2])]

        query = tf.concat(outputs[::3], axis=0)
        key = tf.concat(outputs[1::3], axis=0)
        value = tf.concat(outputs[2::3], axis=0)

        assert query.shape == value.shape == key.shape, \
            'Invalid shape'

        scores = self._attention(
            query=query, key=key, value=value, mask=mask)
        scores = tf.reshape(scores, (-1, self.d_model))

        return self._output(scores)


class PositionWiseFFN(K.layers.Layer):

    def __init__(self, d_model: int = 512, dff: int = 2048):
        super(PositionWiseFFN, self).__init__()

        self._ffn = K.Sequential(
            [create_linear(units=dff, activation=tf.nn.relu),
             create_linear(units=d_model)])

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return self._ffn(inputs)


class AddAndLayerNorm(K.layers.Layer):

    def __init__(self, dropout_prob: float = 0.1,
                 is_training: bool = True):
        super(AddAndLayerNorm, self).__init__()
        self.is_training = is_training

        epsilon = 1E-6
        self._layer_norm = K.layers.LayerNormalization(
            epsilon=epsilon)

        self._dropout = K.layers.Dropout(dropout_prob) \
            if dropout_prob else None

    def build(self, input_shape):
        pass

    def call(self, x, attn):
        if self._dropout is not None:
            y = x + self._dropout(attn, self.is_training)
        return self._layer_norm(y)


class Encoder(K.layers.Layer):

    def __init__(self, **kwargs: dict):
        super(Encoder, self).__init__()

        d_model = kwargs.get('d_model', 512)
        h = kwargs.get('h', 8)
        dff = kwargs.get('dff', 2048)
        dropout_prob = kwargs.get('dropout_prob', 0.1)

        self._mha = MultiHeadAttention(h=h, d_model=d_model)
        self._alm1 = AddAndLayerNorm(dropout_prob=dropout_prob)
        self._ffn = PositionWiseFFN(d_model=d_model, dff=dff)
        self._alm2 = AddAndLayerNorm(dropout_prob=dropout_prob)

    def build(self, input_shape):
        pass

    def call(self, inputs, masks):
        x = self._alm1(inputs[0],
                       self._mha([inputs] * 3, mask=masks))
        return self._alm2(x, self._ffn(x))


class Decoder(K.layers.Layer):

    def __init__(self, **kwargs: dict):
        super(Decoder, self).__init__()

        d_model = kwargs.get('d_model', 512)
        h = kwargs.get('h', 8)
        dff = kwargs.get('dff', 2048)
        dropout_prob = kwargs.get('dropout_prob', 0.1)

        self._mmha = MultiHeadAttention(h=h, d_model=d_model)
        self._alm1 = AddAndLayerNorm(dropout_prob=dropout_prob)
        self._mha = MultiHeadAttention(h=h, d_model=d_model)
        self._alm2 = AddAndLayerNorm(dropout_prob=dropout_prob)
        self._ffn = PositionWiseFFN(d_model=d_model, dff=dff)
        self._alm3 = AddAndLayerNorm(dropout_prob=dropout_prob)

    def build(self, input_shape):
        pass

    def call(self, m, x, x_m, y_m):
        """
        m: memory
        x: source
        x_m: source mask
        y_m: target mask
        """
        x = self._alm1(x, self._mmha([x] * 3, y_m))
        x = self._alm2(x, self._mha([x, m, m], x_m))
        return self._alm3(x, self._ffn(x))


class PositionEncoding(K.layers.Layer):

    def __init__(self, **kwargs: dict):
        super(PositionEncoding, self).__init__()

        self.d_model = kwargs.get('d_model', 512)
        max_len = kwargs.get('max_lenght', 5000)
        dropout_prob = kwargs.get('dropout_prob', 0.1)
        self.is_training = kwargs.get('is_training', True)

        self._dropout = K.layers.Dropout(dropout_prob) \
            if dropout_prob else None

        positions = np.arange(0, max_len,
                              dtype=np.float32)[:, np.newaxis]
        dimensions = np.arange(0, self.d_model,
                               dtype=np.float32)[np.newaxis, :]

        encoding = positions / (10000 ** ((2 * i) / self.d_model))
        encoding[:, 0::2] = np.sin(encoding[:, 0::2])
        encoding[:, 1::2] = np.cos(encoding[:, 1::2])
        self.encoding = tf.convert_to_tensor(encoding)

    def call(self, inputs):
        x = inputs + self.encoding
        return self._dropout(x, self.is_training) \
            if self._dropout is not None else x


class Transformer(K.Model):

    def __init__(self, output_shape, **kwargs: dict):
        super(Transformer, self).__init__()

        encoder_stacks = kwargs.get('encoder_stacks', 6)
        decoder_stacks = kwargs.get('decoder_stacks', 6)

        self._encoders = [Encoder(**kwargs)
                          for _ in range(encoder_stacks)]
        self._decoders = [Decoder(**kwargs)
                          for _ in range(decoder_stacks)]

        self._linear = create_linear(units=output_shape,
                                     activation=tf.nn.softmax)

    def call(self, x, y, x_m, y_m):
        """
        Args:
        x: source
        y: target
        x_m: source mask
        y_m: target_mask
        """
        for encoder in self._encoders:
            x = encoder(x, x_m)

        for decoder in self._decoders:
            x = decoder(x, y, x_m, y_m)
        return self._linear(x)


if __name__ == '__main__':

    d_model = 512
    word = tf.random.uniform([2, d_model])
    # word = tf.random.uniform((1, 60, 512))

    t = Transformer(output_shape=8000)
    t(word)
