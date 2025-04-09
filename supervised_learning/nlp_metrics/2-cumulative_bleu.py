#!/usr/bin/env python3
"""Cumulative N-gram BLEU score Module"""
import math


def cumulative_bleu(references, sentence, n):
    """Function that calculates the cumulative n-gram BLEU score for a
    sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    n is the size of the largest n-gram to use for evaluation
    All n-gram scores should be weighted evenly
    Returns: the cumulative n-gram BLEU score

    """
    # each n-gram level gets equal weight
    weights = [1 / n] * n
    scores = []

    for i in range(1, n + 1):
        # create n-grams
        s_ngrams = [
            tuple(sentence[j:j + i])
            for j in range(len(sentence) - i + 1)
        ]
        # count how often each n-gram appears
        s_counts = {ng: s_ngrams.count(ng) for ng in set(s_ngrams)}

        # clip counts based on the max
        clip = {
            ng: min(
                s_counts[ng],
                max(
                    [[
                        tuple(r[j:j + i])
                        for j in range(len(r) - i + 1)].count(ng)
                        for r in references] or [0]
                )
            )
            for ng in s_counts
        }

        # precision for this n-gram level
        precision = sum(clip.values()) / len(s_ngrams) if s_ngrams else 0
        # if precision is 0, smooth it to avoid log(0)
        scores.append(precision if precision > 0 else 1e-8)

    # geometric mean of all n-gram precisions
    geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, scores)))

    # brevity penalty (BP)
    ref_len = min(
        (len(r), abs(len(r) - len(sentence)))for r in references)[0]
    if len(sentence) > ref_len:
        bp = 1
    elif len(sentence) == 0:
        bp = 0
    else:
        bp = math.exp(1 - ref_len / len(sentence))

    # calculate BLEU score
    bleu_score = bp * geo_mean

    # return final cumulative BLEU score
    return bleu_score
