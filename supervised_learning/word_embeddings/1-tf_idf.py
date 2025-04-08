#!/usr/bin/env python3
"""TF-IDF Module"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used
    Returns: embeddings, features

    """
    # create the vectorizer, using the provided vocab
    v = TfidfVectorizer(vocabulary=vocab)

    # learn the vocabulary and compute the TF-IDF matrix
    embeddings = v.fit_transform(sentences).toarray()

    # get the list of features (words used)
    features = v.get_feature_names_out()

    # return the TF-IDF matrix and feature list
    return embeddings, features
