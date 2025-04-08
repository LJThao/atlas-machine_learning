#!/usr/bin/env python3
"""Extract Word2Vec Module"""
import tensorflow as tf


def gensim_to_keras(model):
    """Function that converts a gensim word2vec model to a keras
    Embedding layer:

    model is a trained gensim word2vec models
    Returns: the trainable keras Embedding

    """
    # gets words from the model
    weights = model.wv.vectors

    # creates keras embedding layer
    trainable_embedding = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=True
    )

    return trainable_embedding
