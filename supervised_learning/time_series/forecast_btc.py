#!/usr/bin/env python3
"""Forecast Module"""
import numpy as np
import tensorflow as tf


# load preprocessed data
X = np.load('X.npy')
y = np.load('y.npy')

# create tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

# build the model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=X.shape[1:]),
    tf.keras.layers.Dense(1)
])

# compile with MSE loss
model.compile(optimizer='adam', loss='mse')

# train the model
model.fit(dataset, epochs=10)

# save the model
model.save('btc_forecast_model.h5')
