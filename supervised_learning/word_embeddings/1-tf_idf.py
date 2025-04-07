#!/usr/bin/env python3
"""TF-IDF Module"""


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features

    """
