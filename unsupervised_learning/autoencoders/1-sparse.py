#!/usr/bin/env python3
"""Sparse Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Function that that creates a sparse autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    lambtha is the regularization parameter used for L1 regularization on
    the encoded output
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the sparse autoencoder model

    """
    # build the encoder - compressing the inputs
    encode_input = keras.Input(shape=(input_dims,))
    x = encode_input
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    # add L1 regularization to the latent layer
    x = keras.layers.Dense(
        latent_dims, activation="relu",
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(encode_input, x, name="encoder")

    # build the decoder - reconstructing the original input
    decode_input = keras.Input(shape=(latent_dims,))
    x = decode_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)
    x = keras.layers.Dense(input_dims, activation="sigmoid")(x)
    decoder = keras.Model(decode_input, x, name="decoder")

    # combine encoder + decoder to make the sparse autoencoder
    auto = keras.Model(
        encode_input, decoder(encoder(encode_input)), name="autoencoder"
    )
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
