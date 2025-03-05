#!/usr/bin/env python3
"""Vanilla Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model

    """
    # build the encoder - compressing the inputs
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)

    # build the decoder - reconstructing the original input
    latent = keras.layers.Dense(latent_dims, activation=None)(x)
    encoder = keras.Model(inputs, latent, name='encoder')

    lat_inputs = keras.Input(shape=(latent_dims,))
    x = lat_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)

    x = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(lat_inputs, x, name='decoder')

    # combining the encoder + decoder to make the autoencoder
    auto = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
