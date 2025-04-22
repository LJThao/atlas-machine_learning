#!/usr/bin/env python3
"""Dataset Module"""
import tensorflow_datasets as tfds
import transformers


class Dataset():
    """Loads and preps a dataset for machine translation"""
    def __init__(self):
        """Initializes the dataset"""
        # load dataset
        data, _ = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            as_supervised=True,
            with_info=True
        )
        self.data_train = data['train']
        self.data_valid = data['validation']

        # init tokenizers from pre-trained models
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Function that creates sub-word tokenizers for our dataset

        data is a tf.data.Dataset whose examples are formatted
        as a tuple (pt, en)
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        """
        # convert dataset to python iterable
        pt_corpus = (pt.decode('utf-8') for pt, _ in data.as_numpy_iterator())
        en_corpus = (en.decode('utf-8') for _, en in data.as_numpy_iterator())

        # build the tokenizer for Portuguese and English
        tokenizer_pt = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                pt_corpus,
                target_vocab_size=2**13
            )
        )
        tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                en_corpus,
                target_vocab_size=2**13
            )
        )
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Function that encodes a translation into token IDs"""
        # encoding the Portuguese and English sentences with token IDs
        pt = pt.decode('utf-8')
        en = en.decode('utf-8')

        pt_tokens = [
            self.tokenizer_pt.vocab_size,
            *self.tokenizer_pt.encode(pt),
            self.tokenizer_pt.vocab_size + 1
        ]
        en_tokens = [
            self.tokenizer_en.vocab_size,
            *self.tokenizer_en.encode(en),
            self.tokenizer_en.vocab_size + 1
        ]
        return pt_tokens, en_tokens