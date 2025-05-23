#!/usr/bin/env python3
"""Convolutional Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Function that creates a convolutional autoencoder:

    input_dims is a tuple of integers containing the dimensions of
    the model input
    filters is a list containing the number of filters for each convolutional
    layer in the encoder, respectively
    the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of
    the latent space representation
    Each convolution in the encoder should use a kernel size of (3, 3)
    with same padding and relu activation, followed by max pooling of
    size (2, 2)
    Each convolution in the decoder, except for the last two,
    should use a filter size of (3, 3) with same padding and relu activation,
    followed by
    upsampling of size (2, 2)
    The second to last convolution should instead use valid padding
    The last convolution should have the same number of filters as the
    number of channels in input_dims with sigmoid activation and no upsampling
    Returns: encoder, decoder, auto
    encoder is the encoder model
    decoder is the decoder model
    auto is the full autoencoder model

    """
    # build the encoder then extracting the features
    encode_input = keras.Input(shape=input_dims)
    x = encode_input
    for f in filters:
        x = keras.layers.Conv2D(
            f, kernel_size=(3, 3), padding="same", activation="relu"
        )(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
    encoder = keras.Model(encode_input, x, name="encoder")

    # build the decoder - reconstructing the original image
    decode_input = keras.Input(shape=latent_dims)
    x = decode_input

    x = keras.layers.Conv2D(filters[-1], (3, 3),
                            padding="same", activation="relu")(x)
    x = keras.layers.UpSampling2D((2, 2))(x)

    for f in filters[:1:-1]:
        x = keras.layers.Conv2D(f, (3, 3),
                                padding="same", activation="relu")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)

    x = keras.layers.Conv2D(
        filters[0], kernel_size=(3, 3), padding="valid", activation="relu"
    )(x)

    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(input_dims[-1], kernel_size=(3, 3),
                            padding="same", activation="sigmoid")(x)

    decoder = keras.Model(decode_input, x, name="decoder")

    # connects the encoder + decoder into autoencoder
    auto = keras.Model(
        encode_input, decoder(encoder(encode_input)), name="autoencoder"
    )
    auto.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, auto
