#!/usr/bin/env python3
"""N-gram BLEU score Module"""
import math


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the n-gram to use for evaluation
    Returns: the n-gram BLEU score

    """
    # generate n-grams
    s_ngrams = [
        tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)
    ]
    s_counts = {ng: s_ngrams.count(ng) for ng in set(s_ngrams)}

    # count clipped matches
    clipped = {}
    for ng in s_counts:
        max_count = max(
            [
                [tuple(r[i:i + n]) for i in range(len(r) - n + 1)].count(ng)
                for r in references
            ]
        )
        clipped[ng] = min(s_counts[ng], max_count)

    # calculate precision
    precision = sum(clipped.values()) / len(s_ngrams) if s_ngrams else 0

    # brevity penalty
    ref_len = min(
        (len(r), abs(len(r) - len(sentence)))
        for r in references
    )[0]

    if len(sentence) == 0:
        bp = 0
    elif len(sentence) > ref_len:
        bp = 1
    else:
        bp = math.exp(1 - ref_len / len(sentence))

    # calculate the n-gram BLEU score
    bleu_score = bp * precision

    return bleu_score
