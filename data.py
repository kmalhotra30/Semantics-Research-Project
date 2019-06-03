"""
This file implements the datasets used for discourse modelling
for the task of metaphor detection.

It includes:
- TextDataset: general dataset with functionality to load vocabularies,
  create batches and dataloaders.
  to prepare batches (mapping words and targets to tensors).
- MetaphorDataset: TextDataset instantiation containing functionality to load
  the VUA metaphor corpus.

Author: Verna Dankers
"""
import csv
import random
import codecs
import itertools
import logging
import numpy as np

from collections import defaultdict
from os.path import join as join
from torch.utils.data import Dataset
from sample import Sample, Batch


class TextDataset(Dataset):
    def __init__(self, config, mode, w2i, i2w):
        super().__init__()
        """
        Args:
            config (dict): contains command line arguments
            mode (str): attention | metaphor | emotion
            w2i (dict): maps words to indices
            i2w (dict): maps indices to words
        """
        self.batch_size = 64
        self.w2i = w2i
        self.i2w = i2w
        self.vocab = list(w2i.keys())
        self.model_name = config["model_type"]
        self.batch_id = 0
        self.sort_data = config["sort_data"]

    def to_batch(self, samples):
        """
        Given a list of examples (each from __getitem__),
        combine them to form a single batch by padding.

        Returns:
        -------
        batch_padded_example_text: LongTensor
          LongTensor of shape (batch_size, longest_sequence_length) with the
          padded text for each example in the batch.
        length: LongTensor
          LongTensor of shape (batch_size,) with the unpadded length of the example.
        example_label: LongTensor
          LongTensor of shape (batch_size,) with the label of the example.
        """
        batch = Batch()
        # Get the length of the longest sequence in the batch
        max_length = max([sample.max_length for sample in samples])

        for sample in samples:
            # Focus sentence
            amount_to_pad = max_length - len(sample.label_seq)
            # Prepare the labels to compute the losses
            batch.label_seqs.append(sample.label_seq  + [-1]*amount_to_pad)
            batch.discourse_lengths.append(sample.discourse_length)
            batch.focus_positions.append(sample.focus_position)

            for sentence in sample.discourse:
                amount_to_pad = max_length - len(sentence)
                sample_indices = [self.w2i[w if w in self.w2i else "UNK"] for w in sentence]
                batch.idx.append(sample_indices + [self.w2i["PAD"]] * amount_to_pad) # Map words to indices for the Glove embeddings
                batch.words.append(sentence) # Collect the words for ELMO
                batch.lengths.append(len(sentence)) # Collect the lengths needed for packed padded sequence in RNN
        batch.lengths = [max(x, 1) for x in batch.lengths]

        batch.to_tensor()
        batch.to_cuda()
        return batch

    def dataloader(self, dataset, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        for i in range(0, len(dataset), batch_size):
            yield self.to_batch(dataset[i:i+batch_size])

    def get_train(self):
        batch_id = self.batch_id
        self.batch_id += self.batch_size
        if self.batch_id > len(self.train) - 1:
            self.batch_id = 0
            random.shuffle(self.train)
        batch = self.train[self.batch_id: self.batch_id+self.batch_size]
        return self.to_batch(batch)



class MetaphorDataset(TextDataset):
    def __init__(self, config, w2i, i2w, rep):
        super().__init__(config, "metaphor", w2i, i2w)
        """
        Get raw dataset as a list:
          Each element is a triple:
            a sentence: string
            a index: int: idx of the focus verb
            a label: int 1 or 0

        :param config: argument line configurations
        """

        self.task = "labeling"
        self.batch_size = config["batch_size"]
        self.context_window = config["context_window"]
        self.discourse = defaultdict(lambda : dict())

        # Load train, validation and test data
        self.train = self.load(join(config["meta_folder"], config["meta_train"]))
        self.validation = self.load(join(config["meta_folder"], config["meta_valid"]))
        self.test = self.load(join(config["meta_folder"], config["meta_test"]))

        # Create dataloaders that give you batches during training and evaluating
        self.train_loader = itertools.cycle(self.dataloader(self.train))
        self.test_loader = self.dataloader(self.test)
        self.validation_loader = self.dataloader(self.validation)
        self.vocab = list(self.vocab)
        self.sort_data = config["sort_data"]
        print(self.sort_data)

    def get_discourse(self, focus, text, sentence):
        discourse = []
        rest = []
        for i in range(sentence - self.context_window, sentence + self.context_window + 1):
            if i in self.discourse[text]:
                discourse.append(self.discourse[text][i])
            else:
                rest.append([])
        focus_position = discourse.index(focus)
        discourse.extend(rest)
        return discourse, focus_position

    def load(self, filename):
        """
        Load Metaphor Dataset from CSV file..

        :param filename: name of the dataset file
        :returns samples: list containing Sample objects
        """
        samples = []
        raw_samples = csv.DictReader(codecs.open(filename, 'r', 'latin-1'))

        for sample in raw_samples:
            if "a" in sample["sentence_id"] or len(sample["sentence_txt"].split()) <= 1:
                continue
            sentence = sample["sentence_txt"].replace("M_", "").replace("L_", "").lower().split()
            self.discourse[sample["txt_id"]][int(sample["sentence_id"])] = sentence
            sample = Sample(sentence=sample["sentence_txt"], text_id=sample["txt_id"],
                            sent_id=sample["sentence_id"])
            samples.append(sample)

        for sample in samples:
            discourse, focus_position = self.get_discourse(sample.sentence, sample.text_id, sample.sent_id)
            sample.update_discourse(discourse, focus_position)

        if self.sort_data != 0:
            samples = sorted(samples, key=lambda x: x.max_length)
        return samples
