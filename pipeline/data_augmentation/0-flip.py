#!/usr/bin/env python3
"""Flip Module"""
import tensorflow as tf


def flip_image(image):
    """Function that flips an image horizontally:

    image is a 3D tf.Tensor containing the image to flip
    Returns the flipped image

    """
    return tf.image.flip_left_right(image)
