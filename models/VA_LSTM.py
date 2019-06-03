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


class VA_LSTM(Model):
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
        super(VA_LSTM, self).__init__(glove_filename, elmo_weights, elmo_options)
        self.name = "VA_LSTM"
        self.context_window = context_window
        self.hidden_size = hidden_size
        direc = 2 # bidirectional lstm

        # Dropout layer
        self.dropout_embs = nn.Dropout(dropout1)
        self.dropout_linear = nn.Dropout(dropout3)

        # Set up the RNN: use an LSTM here.
        self.rnn = self.initialize_lstm(self.embedding_dim, hidden_size, num_layers, dropout2)
        #self.output_metaphor = nn.Linear(hidden_size * direc, 2)
        self.output_metaphor = nn.Linear(hidden_size * direc * 2, 2)

        self.attention_linear = nn.Linear(hidden_size * direc, 1)

    def forward(self, batch):
        # 1. Retrieve embeddings + add dropout

        focus_sentence = self.dropout_embs(self.embed(batch.idx, batch.words))
        #print('FO',focus_sentence.size(), type(focus_sentence), focus_sentence)
        # 2. Run embeddings through RNN
        output = self.run_rnn(focus_sentence, batch, self.rnn)
        #print('OO',output.size(), type(output), output)
        #input_encoding = self.dropout_linear(output)
        input_encoding = output

        # Determine the shape of the 3d tensor used for discourse
        batch_size, sentence_length = batch.idx.shape

        context_size = self.context_window * 2 + 1

        shape = (int(batch_size/context_size), context_size, sentence_length, self.hidden_size * 2)

        #Reshaping Input Encoding to B X CS X MAX_Sent_Len X hid *2 
        input_encoding = input_encoding.view(shape)

        shape_mask = (int(batch_size/context_size), context_size)

        shape_attention = (int(batch_size/context_size), context_size * sentence_length, self.hidden_size * 2)

        #Mask Length is a tensor of shape B X CS. It contains the lengths of sentences. Used for masking attention weights.
        mask_length = batch.lengths.view(shape_mask)

        #mask_attention is a tensor which will store the binary masks (0 or 1)
        mask_attention = torch.empty(int(batch_size/context_size), context_size*sentence_length).cuda()

        for j in range(int(batch_size/context_size)):

            #Masks for each sentence
            mask_list_ = [[1]*int(mask_length[j, i]) + [0]*(sentence_length - int(mask_length[j, i])) for i in range(context_size)]

            mask_list_ = [item for sublist in mask_list_ for item in sublist]

            #Adding binary masks 1/0 to the tensor.
            mask_attention[j, :] = torch.FloatTensor(mask_list_).squeeze()


        attention_input_encoding = input_encoding.view(shape_attention)

        #Applying Linear layer for attention
        attention_activation = self.attention_linear(attention_input_encoding).squeeze()

        #Masking the activations before softmax
        masked_attention_activation = attention_activation * mask_attention

        #Applying softmax non linearity
        masked_attention_normalized = F.softmax(masked_attention_activation, dim = 1)
        
        #Trial
        masked_attention_normalized = masked_attention_normalized * mask_attention

        #Multiplying attention weights with  Bi-LSTM Activations

        attention_weighted_encoding = masked_attention_normalized.unsqueeze(2) * attention_input_encoding

        #Computing Discourse representation

        discourse_representation = attention_weighted_encoding.sum(1)

        discourse_representation_repeated = discourse_representation.repeat(sentence_length, 1, 1).transpose(0,1)

        #input_encoding = input_encoding[:, batch.focus_positions, :, :][:, 0, :, :]
        input_encoding = input_encoding[range(len(input_encoding)), batch.focus_positions]

        #Concat of discourse + BI-LSTM activation of words
        discourse_input_concat = torch.cat((discourse_representation_repeated, input_encoding), 2)

        #unnormalized_output = self.output_metaphor(input_encoding)
        discourse_input_concat = self.dropout_linear(discourse_input_concat)
        unnormalized_output = self.output_metaphor(discourse_input_concat)

        return torch.log_softmax(unnormalized_output, dim=-1)
