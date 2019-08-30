"""
This file implements an abstract Model class with functionalities used
by other models listed in the same folder.

Author: Verna Dankers
"""
import pickle
import torch
import torch.nn as nn
import logging

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as pad
from allennlp.nn.util import sort_batch_by_length, masked_softmax
from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np


class Model(nn.Module):
    def __init__(self, glove_filename, elmo_weights, elmo_options, use_elmo = 1):
        super().__init__()

        self.use_elmo = use_elmo

        if self.use_elmo == 1:
            self.embedding_dim = 1324
        else:
            self.embedding_dim = 768
                 
        self.elmo = Elmo(elmo_options, elmo_weights, 1,
                         requires_grad=False, dropout=0)

        glove_file = open(glove_filename, 'rb')
        embeddings = torch.FloatTensor(pickle.load(glove_file))
        self.vocab_size, _ = embeddings.shape
        self.embeddings = nn.Embedding.from_pretrained(embeddings)

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        if torch.cuda.is_available():
            self.bert = self.bert.cuda()

    def initialize_lstm(self, embedding_dim, hidden_size, num_layers, dropout2):
        rnn = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout2, batch_first=True,
            bidirectional=True
        )
        return rnn

    def embed(self, indices, words):

        glove = self.embeddings(indices)

        if self.use_elmo == 1:
            elmo = self.retrieve_elmo(words)
            return torch.cat((glove, elmo), dim=-1)
        else:
            bert = self.retrieve_bert(words)

            if torch.cuda.is_available():
                bert = bert.cuda()
            return bert

            #return torch.cat((glove, bert), dim=-1)

    def retrieve_bert(self, words):

        self.bert.eval()
        
        sentences = ['[CLS] ' + " ".join(sent) for sent in words]
        max_len_words = np.max([len(w) for w in words])

        bert_tokeinzed = [self.bert_tokenizer.tokenize(sent) for sent in sentences]

        input_ids = [self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in bert_tokeinzed]
        max_len = np.max([len(idx) for idx in input_ids])

        attention_mask = [[1] * len(idx) + [0] * (max_len - len(idx)) for idx in input_ids]
        attention_mask = torch.LongTensor(attention_mask)

        input_ids_padded = [idx + [0] * (max_len - len(idx)) for idx in input_ids]
        input_ids_padded = torch.LongTensor(input_ids_padded)

        if torch.cuda.is_available():
            input_ids_padded = input_ids_padded.cuda()
            attention_mask = attention_mask.cuda()

        all_tokens, pooled = self.bert(input_ids_padded, token_type_ids = None, attention_mask = attention_mask,
                                      output_all_encoded_layers=False)

        mask_subword = [[0] + [1] * (len(b) - 1) for b in bert_tokeinzed]


        # for id, t in enumerate(bert_tokeinzed):
        #     for idx in range(len(t) - 1):
        #         for idx2 in range(idx + 1, len(t)):
        #             if t[idx2][0].isalpha() == False:
        #                 mask_subword[id][idx2] = 0
        #             else:
        #                 break

        for id, t in enumerate(bert_tokeinzed):
            for idx,token in enumerate(t):

                if t[idx][0].isalpha() == False or len(t[idx]) <= 2:
                    mask_subword[id][idx] = 0


        batch_size = all_tokens.size(0)
        #max_length = all_tokens.size(1)
        embed_size = all_tokens.size(2)
        tokens_tensor = torch.torch.FloatTensor(batch_size, max_len_words, embed_size).uniform_(-0.05, 0.05)

        for idx, data_pt in enumerate(all_tokens):

            indice = 0
            for length in range(len(bert_tokeinzed[idx])):

                if mask_subword[idx][length] == 1:

                    try:
                        tokens_tensor[idx][indice] = all_tokens[idx][length]
                        indice += 1
                    except:
                        print(bert_tokeinzed[idx])
                        print(mask_subword[idx])
                else:
                    continue
        return tokens_tensor


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
