#!/usr/bin/env python3
"""Semantic Search Module"""
import os
import numpy as np
import tensorflow_hub as hub


def semantic_search(corpus_path, sentence):
    """Function that performs semantic search on a corpus of documents:

    corpus_path is the path to the corpus of reference documents on which
    to perform semantic search
    sentence is the sentence from which to perform semantic search
    Returns: the reference text of the document most similar to sentence

    """
    # list to hold the input sentence and reference documents
    texts = [sentence]

    # load reference documents
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            file_path = os.path.join(corpus_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                texts.append(f.read())

    # load Universal Sentence Encoder model
    model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-large/5")
    embeddings = model(texts)

    # compute cosine similarities between input sentence and each document
    input_vector = embeddings[0]
    similarities = [
        np.dot(input_vector, doc_vec) /
        (np.linalg.norm(input_vector) * np.linalg.norm(doc_vec))
        for doc_vec in embeddings[1:]
    ]

    # find the most similar document
    reference_task = texts[np.argmax(similarities) + 1]
    return reference_task
