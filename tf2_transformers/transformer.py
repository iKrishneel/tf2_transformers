#!/usr/bin/env python

import numpy as np

import tensorflow as tf
from tensorflow import keras as K


def create_linear(self, **kwargs):
    return K.layers.Dense(**kwargs)


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

        scores = tf.matmul(query, key, transpose_b=True)
        scores /= np.sqrt(query.shape[-1])
        
        if mask is not None:
            # todo: mask
            pass
        
        attn_score = K.activations.softmax(scores, axis=-1)
        attn_score = self.dropout(attn_score, self.is_training) \
          if self.dropout else attn_score

        self._attn_score = attn_score
        return tf.matmul(attn_score, value, name='scores')

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

    def call(self, inputs: list, mask = None):
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
        assert num_layers > 0, 'Number of layers must be positive'

        self._ffn = K.K.Sequential(
            [create_linear(units=d_model, activation=tf.nn.relu),
             create_linear(units=dff)])

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return self._ffns(inputs)


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
    
    def call(self, inputs):
        x = self._alm1(inputs, self._mha(inputs))
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
        self._mha = MultiHeadAttention(h=h, d_model=d_model),
        self._alm2 = AddAndLayerNorm(dropout_prob=dropout_prob)       
        self._ffn = PositionWiseFFN(d_model=d_model, dff=dff)
        self._alm3 = AddAndLayerNorm(dropout_prob=dropout_prob)
        
    def build(self, input_shape):
        pass

    def call(self, attn_inp, shifted_inp):
        x = self._alm1(shifted_inp, self._mmha(shifted_inp))
        x = self._alm2(x, self._mha([attn_inp, attn_inp, x]))
        x = self._alm3(x, self._ffn(x))

        
 class Transformer(K.Model):

    def __init__(self, output_shape, **kwargs: dict):
        super(Transformer, self).__init__()

        encoder_stacks = kwargs.get('encoder_stacks', 6)
        decoder_stacks = kwargs.get('decoder_stacks', 6)

        self._encoders = K.Sequential(
            [Encoder(**kwargs) for _ in range encoder_stacks])
        self._decoders = K.Sequential(
            [Decoder(**kwargs), for _ in range decoder_stacks])
        
        self._linear = create_linear(units=output_shape,
                                     activation=tf.nn.softmax)
        
    def call(self, x, o):
        x = self.decoder_stacks(self._encoders(x), o)
        return self._linear(x)
    

if __name__ == '__main__':

    d_model = 512
    word = tf.random.uniform([1, d_model])

    m = MultiHeadAttention(h=8, d_model=d_model)
    m.build(input_shape=[d_model] * 3)
    
    x = m([word] * 3)
    
    print(m.count_params())
    
    print(x.shape)
    
    # model = K.Model(inputs=[word, word, word], outputs = m)

    # print(m.summary())
    # m([word, word, word])
    
