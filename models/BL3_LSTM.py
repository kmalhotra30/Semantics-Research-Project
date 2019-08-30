"""
This file implements an LSTM that can be used
for joint learning of metaphor detection and emotion classification.
The LSTM encoder is shared, but is followed by task-specific classification
layers.

Author: Verna Dankers
"""
import math
import pickle
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F

from allennlp.nn.util import masked_softmax

from .Model import Model


class BL3_LSTM(Model):
    """
    Model implementing LSTMs followed by (task-specific) classification layers,
    that can be trained for multiple tasks at the same time.
    """
    def __init__(self, glove_filename, elmo_options, elmo_weights, hidden_size=200,
                 dropout1=0.5, dropout2=0.0, dropout3=0.1, num_layers=1, context_window=1,
                 **kwargs):
        """
        Args:
            hidden_size (int): The size of the RNN hidden state.
            glove_filename (str): filename of pickled glove vectors
            dropout1 (float): dropout on input to RNN
            dropout2 (float): dropout in RNN
            dropout3 (float): dropout on hidden state of RNN to linear layer
        """
        super(BL3_LSTM, self).__init__(glove_filename, elmo_weights, elmo_options, use_elmo=kwargs['use_elmo'])
        self.name = "BL3_LSTM"
        self.context_window = context_window
        self.hidden_size = hidden_size
        direc = 2 # bidirectional lstm

        # Dropout layer
        self.dropout_embs = nn.Dropout(dropout1)
        self.dropout_linear = nn.Dropout(dropout3)

        # Set up the RNN: use an LSTM here.
        self.rnn = self.initialize_lstm(self.embedding_dim, hidden_size, num_layers, dropout2)
        self.output_metaphor = nn.Linear(hidden_size * direc, 2)

    def forward(self, batch):
        # 1. Retrieve embeddings + add dropout

        focus_sentence = self.dropout_embs(self.embed(batch.idx, batch.words))
        
        # 2. Run embeddings through RNN
        output = self.run_rnn(focus_sentence, batch, self.rnn)
        input_encoding = self.dropout_linear(output)

        # Determine the shape of the 3d tensor used for discourse
        batch_size, sentence_length = batch.idx.shape
        context_size = self.context_window * 2 + 1
        shape = (int(batch_size/context_size), context_size, sentence_length, self.hidden_size * 2)

        input_encoding = input_encoding.view(shape)
        #input_encoding = input_encoding[:, batch.focus_positions, :, :][:, 0, :, :]
        input_encoding = input_encoding[range(len(input_encoding)), batch.focus_positions]
        unnormalized_output = self.output_metaphor(input_encoding)
        return torch.log_softmax(unnormalized_output, dim=-1)
