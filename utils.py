#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences


def build_lstm(units):
    """Build a LSTM model with the given number of units in its cell."""
    inputs = Input((None, 1))  # Arbitrary length sequences x 1 feature
    lstm   = LSTM(units)(inputs)
    out    = Dense(1, activation='sigmoid')(lstm)
    model  = Model(inputs=inputs, outputs=out)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


def xor(array):
    """Compute the xor of a binary array using parity."""
    return sum(array) % 2


def labels(data):
    """Get the labels for a XOR dataset."""
    return np.apply_along_axis(xor, 1, data)


def random_binary_data(n, *, min_len, max_len):
    """Get random data for training."""
    if min_len == max_len:
        data = np.random.choice([0, 1], (n, min_len, 1))
    else:
        data = [
            np.random.choice([0, 1], (np.random.randint(min_len, max_len), 1))
            for i in range(n)
        ]
        data = pad_sequences(data, maxlen=max_len, dtype='int32',
                             padding='pre', truncating='post', value=0.)

    return data


def compute_accuracy(model, x, y):
    """Get the accuracy of a model over x and y."""
    return sum(np.round(model.predict(x)) == y) / len(y)
