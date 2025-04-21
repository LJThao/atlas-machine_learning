#!/usr/bin/env python3
"""Dataset Module"""
import tensorflow_datasets as tfds
import transformers


class Dataset():
    """Loads and preps a dataset for machine translation"""
    def __init__(self):
        """Initializes the dataset"""
        # load dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # init tokenizers from pre-trained models
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """Function that creates sub-word tokenizers for our dataset

        data is a tf.data.Dataset whose examples are formatted as a tuple (pt, en)
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased")
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased")

        return tokenizer_pt, tokenizer_en