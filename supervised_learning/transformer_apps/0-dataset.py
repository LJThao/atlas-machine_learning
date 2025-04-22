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
        # separate sets
        self.data_train = data['train']
        self.data_valid = data['validation']

        # build tokenizers from training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        # print if loaded
        print("Datasets correctly loaded")

    def tokenize_dataset(self, data):
        """Function that creates sub-word tokenizers for our dataset

        data is a tf.data.Dataset whose examples are formatted
        as a tuple (pt, en)
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        """
        # convert datasets
        pt_corpus = (pt.decode('utf-8') for pt, _ in data.as_numpy_iterator())
        en_corpus = (en.decode('utf-8') for _, en in data.as_numpy_iterator())

        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        ).train_new_from_iterator(pt_corpus, vocab_size=2**13)

        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        ).train_new_from_iterator(en_corpus, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Function that encodes a translation into token IDs"""
        # decoding bytes to strings
        pt = pt.decode('utf-8')
        en = en.decode('utf-8')

        # adding the tokens to the encoded sentences
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
