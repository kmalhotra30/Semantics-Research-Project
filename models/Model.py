"""
This file implements an abstract Model class with functionalities used
by other models listed in the same folder.

Author: Verna Dankers
"""
import pickle
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from allennlp.nn.util import sort_batch_by_length, masked_softmax
from allennlp.modules.elmo import Elmo, batch_to_ids


class Model(nn.Module):
    def __init__(self, glove_filename, elmo_weights, elmo_options):
        super().__init__()

        self.embedding_dim = 1324
                 
        self.elmo = Elmo(elmo_options, elmo_weights, 1,
                         requires_grad=False, dropout=0)

        glove_file = open(glove_filename, 'rb')
        embeddings = torch.FloatTensor(pickle.load(glove_file))
        self.vocab_size, _ = embeddings.shape
        self.embeddings = nn.Embedding.from_pretrained(embeddings)

    def initialize_lstm(self, embedding_dim, hidden_size, num_layers, dropout2):
        rnn = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout2, batch_first=True,
            bidirectional=True
        )
        return rnn

    def embed(self, indices, words):
        glove = self.embeddings(indices)
        elmo = self.retrieve_elmo(words)
        return torch.cat((glove, elmo), dim=-1)

    def retrieve_elmo(self, words):
        """
        Retrieve ELMo embeddings based on lists of words.

        Args:
            words (list): list of lists containing strings

        Returns:
            torch.FloatTensor: ELMo embeddings
        """
        idx = batch_to_ids(words)
        if torch.cuda.is_available():
            idx = idx.cuda()
        return self.elmo(idx)['elmo_representations'][0]

    def run_rnn(self, embedded_input, batch, rnn):
        """
        Run embeddings through RNN and return the output.

        Args:
            embedded_input (torch.FloatTensor): batch x seq x dim
            batch (Batch): batch object containing .lengths tensor
            param (torch.nn.LSTM): LSTM to run the embeddings through

        Returns:
            torch.FloatTensor: hidden states output of LSTM, batch x seq x dim
        """
        (sorted_input, sorted_lengths, input_unsort_indices, _) = \
            sort_batch_by_length(embedded_input, batch.lengths)
        packed_input = pack(sorted_input, sorted_lengths.data.tolist(),
                            batch_first=True)
        rnn.flatten_parameters()
        packed_sorted_output, _ = rnn(packed_input)
        sorted_output, _ = pad(packed_sorted_output, batch_first=True)
        return sorted_output[input_unsort_indices]
