#!/usr/bin/env python3
"""Question Answering Module"""
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer


def question_answer(question, reference):
    """Function that finds a snippet of text within a reference document to
    answer a question:

    question is a string containing the question to answer
    reference is a string containing the reference document from which to
    find the answer
    Returns: a string containing the answer
    If no answer is found, return None
    Your function should use the bert-uncased-tf2-qa model from the
    tensorflow-hub library
    Your function should use the pre-trained BertTokenizer,
    bert-large-uncased-whole-word-masking-finetuned-squad, from the
    transformers library

    """
    # load resources
    tokenizer = BertTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad")
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # tokenize input using tokenizer API
    q_tokens = tokenizer.tokenize(question)
    r_tokens = tokenizer.tokenize(reference)

    # special token formatting
    tokens = ["[CLS]"] + q_tokens + ["[SEP]"] + r_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # segment and attention masks
    q_len = len(q_tokens) + 2
    segment_ids = [0] * q_len + [1] * (len(r_tokens) + 1)
    attention_mask = [1] * len(input_ids)

    # convert to tensors
    ids = tf.constant([input_ids])
    mask = tf.constant([attention_mask])
    segments = tf.constant([segment_ids])

    # predict start and end
    start_idx = tf.argmax(model([ids, mask, segments])[0][0][1:]) + 1
    end_idx = tf.argmax(model([ids, mask, segments])[1][0][1:]) + 1

    # handle invalid span
    if start_idx >= end_idx:
        return None

    # set decoded_text
    decoded_text = tokenizer.convert_tokens_to_string(
        tokens[start_idx:end_idx + 1]
        ).strip()

    # return decoded text
    return decoded_text
