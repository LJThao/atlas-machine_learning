#!/usr/bin/env python3
"""Variational Autoencoder Module"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Function that creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
    the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
    encoder is the encoder model, which should output the latent
    representation, the mean, and the log variance, respectively
    decoder is the decoder model
    auto is the full autoencoder model

    """
    def sample_z(args):
        """Function that transforms a std normal random var into a sample"""
        mean, log_var = args
        epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation="relu")(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sample_z)([mean, log_var])

    encoder = keras.Model(inputs, [z, mean, log_var], name="encoder")

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation="relu")(x)

    outputs = keras.layers.Dense(input_dims, activation="sigmoid")(x)

    decoder = keras.Model(latent_inputs, outputs, name="decoder")

    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs, auto_outputs, name="autoencoder")

    def variational_loss(y_true, y_pred):
        """Function that computes the VAE loss as the sum of reconstruction
        loss and KL divergence. Ensures the latent space follows a std
        normal distribution while minimizing reconstruction error."""
        z_mean, z_log_var = y_pred[1], y_pred[2]

        rec_loss = keras.losses.binary_crossentropy(y_true, y_pred[0])
        rec_loss = keras.backend.sum(rec_loss, axis=-1)

        kl_loss = -0.5 * keras.backend.sum(
            1 + z_log_var
            - keras.backend.square(z_mean)
            - keras.backend.exp(z_log_var),
            axis=-1,
        )

        return keras.backend.mean(rec_loss + kl_loss)

    auto.compile(optimizer="adam", loss=variational_loss)

    return encoder, decoder, auto
