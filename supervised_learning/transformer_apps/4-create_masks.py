#!/usr/bin/env python3
"""Create Masks Module"""
import tensorflow as tf


def create_masks(inputs, target):
    """Function that creates all masks for training/validation:

    inputs is a tf.Tensor of shape (batch_size, seq_len_in) that
    contains the input sentence
    target is a tf.Tensor of shape (batch_size, seq_len_out) that
    contains the target sentence
    This function should only use tensorflow operations in order
    to properly function in the training step
    Returns: encoder_mask, combined_mask, decoder_mask

    """
    # padding mask - 1 where the input == 0
    def pad(x):
        return tf.cast(tf.equal(x, 0), tf.float32)[:, None, None, :]

    # mask for encoder input
    encoder_mask = pad(inputs)

    # padding mask for decoder input
    decoder_pad_mask = pad(target)

    # look-ahead mask to block future tokens
    look_ahead = 1 - tf.linalg.band_part(
        tf.ones((tf.shape(target)[1], tf.shape(target)[1])), -1, 0
    )

    # combine look-ahead with padding
    combined_mask = tf.maximum(
        decoder_pad_mask, look_ahead[tf.newaxis, tf.newaxis, :, :]
    )

    # decoder uses encoder mask again
    decoder_mask = encoder_mask

    return encoder_mask, combined_mask, decoder_mask
