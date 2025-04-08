#!/usr/bin/env python3
"""Unigram BLEU score Module"""
import math


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence:

    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score

    """
    # count how many times the word appears
    counts = {word: sentence.count(word) for word in set(sentence)}

    # keeping the count limiting by clipping
    clipped = {
        word: min(counts[word], max(ref.count(word) for ref in references))
        for word in counts
    }

    # calculate the precision by total matched/total words
    precision = sum(clipped.values()) / len(sentence) if sentence else 0

    # finding the reference length
    closest_ref_len = min(
        (len(ref), abs(len(ref) - len(sentence))) for ref in references
    )[0]

    # applying the brevity penalty
    if len(sentence) > closest_ref_len:
        bp = 1
    elif len(sentence) == 0:
        bp = 0
    else:
        bp = math.exp(1 - closest_ref_len / len(sentence))

    # calculate the BLEU score
    bleu_score = bp * precision

    return bleu_score