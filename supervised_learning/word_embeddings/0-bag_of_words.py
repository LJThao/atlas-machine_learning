#!/usr/bin/env python3
"""Bag of Words Module"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Function that creates a bag of words embedding matrix:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
    If None, all words within sentences should be used

    Returns: embeddings, features
    -> embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
    ->> s is the number of sentences in sentences
    ->> f is the number of features analyzed
    -> features is a list of the features used for embeddings

    You are not allowed to use genism library.

    """
    # tokenize each sentence into lowercase
    token = [re.findall(r"\b\w+(?:'\w+)?\b", s.lower()) for s in sentences]

    # use vocab or we build it from the words
    features = (
        sorted(set(word for sent in token for word in sent))
        if vocab is None else vocab
    )

    # map the word to its index
    word_index = {word: i for i, word in enumerate(features)}

    # init the embedding mat
    s = len(sentences)
    f = len(features)
    embeddings = np.zeros((s, f), dtype=int)

    # iterate over and count the words
    for i, sentence in enumerate(token):
        for word in sentence:
            if word in word_index:
                embeddings[i, word_index[word]] += 1

    # return the embeddings and feature list
    return embeddings, np.array(features)
