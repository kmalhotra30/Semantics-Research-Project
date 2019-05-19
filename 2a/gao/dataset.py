# Importing the Libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import nltk
import math
import re
import torchnlp
import torch
from torchnlp.word_to_vector import GloVe
import pickle
from allennlp.modules.elmo import Elmo, batch_to_ids
nltk.download('punkt')

class DatasetPreProcessorGao:
    
    def __init__(self, seed = 42):
        
        self.seed = seed
        
    def get_datasets(self):
        
        self.original_dataset = {}
        
        for dataset_type in ['train', 'val', 'test']:
            
            self.original_dataset[dataset_type] = pd.read_csv('VUA_seq_formatted_' + dataset_type + '.csv', \
                                                    encoding = 'ISO-8859-1')
            self.original_dataset[dataset_type] = self.original_dataset[dataset_type].dropna()
    
    def pre_process_corpus(self):
        
        self.texts = {}
        
        self.dataset = {}
        
        self.vocab = set()
        
        for dataset_type in ['train', 'val', 'test']:
            
            for index, row in self.original_dataset[dataset_type].iterrows():
                
                text_id = row['txt_id']
                
                sent_id = str(row['sen_ix'])
                
                sentence_txt = row['sentence']
                
                sentence_labels = row['label_seq'][1:-1]
                
                if text_id not in self.texts:
                    self.texts[text_id] = {}
                
                if sent_id not in self.texts[text_id]:
                    self.texts[text_id][sent_id] = {}
                
                self.texts[text_id][sent_id]['token_list'] = sentence_txt.split()
                
                self.vocab = self.vocab.union(set(self.texts[text_id][sent_id]['token_list']))
                
                self.texts[text_id][sent_id]['labels'] = list(map(int, sentence_labels.split(', ')))
                
                assert(len(self.texts[text_id][sent_id]['token_list']) == \
                       len(self.texts[text_id][sent_id]['labels']))
        
        self.w2i = {w : i + 4 for i, w in enumerate(list(self.vocab))}
        self.w2i['<PAD>'] = 0
        self.w2i['<UNK>'] = 1
        self.w2i['<SOS>'] = 2
        self.w2i['<EOS>'] = 3
                
    def get_mappings(self):
        
        self.text_id_mapping = {}
        
        self.text_sent_mapping = {}
        
        text_counter, text_sent_counter = 0, 0
        
        self.text_id_mapping['-1'] = -1
        
        self.text_sent_mapping[('-1', '-1')] = -1
        
        for text_id in self.texts:
            
            for sent_id in self.texts[text_id]:
                
                if text_id not in self.text_id_mapping:
                    
                    self.text_id_mapping[text_id] = text_counter
                    
                    text_counter += 1
                
                if (text_id, sent_id) not in self.text_sent_mapping:
                    
                    self.text_sent_mapping[(text_id, sent_id)] = text_sent_counter
                    
                    text_sent_counter += 1
        
        self.text_id_mapping_reverse = {v : k for k, v in self.text_id_mapping.items()}
        
        self.text_sent_mapping_reverse = {v : k for k, v in self.text_sent_mapping.items()}
    
    def get_context(self, text_id, sent_id, context_window = 0):
    
        temp_context, temp_context_labels, context_sent_ids = {}, {}, {}
    
        temp_context_masked, temp_context_unmasked = [], []
    
        temp_context_labels_masked, temp_context_labels_unmasked = [], []
    
        context_sent_ids_masked, context_sent_ids_unmasked = [], []
    
        (text_str, sent_str) = self.text_sent_mapping_reverse[sent_id]
    
        sent_ids = list(self.texts[text_str])
    
        sent_ids_mapping = [self.text_sent_mapping[(text_str, k)] for k in sent_ids]
    
        sorted_sent_ids = sorted(sent_ids_mapping)
    
        print(sorted_sent_ids)
    
        sent_id_idx = sorted_sent_ids.index(self.text_sent_mapping[(text_str, sent_str)])
    
        print(sent_id_idx)
    
        for i in range(-context_window, context_window + 1):
      
          if sent_id_idx + i < 0 or sent_id_idx + i >= len(sorted_sent_ids):
            
            pass
        
          else:
        
            sent_id_str = self.text_sent_mapping_reverse[sorted_sent_ids[sent_id_idx + i]][1]
        
            if sent_id_idx + i != sent_id_idx:
                temp_context_masked.append(self.texts[text_str][sent_id_str]['token_list'])
                temp_context_labels_masked.append(self.texts[text_str][sent_id_str]['labels'])
                context_sent_ids_masked.append(sent_id_str)
        
            temp_context_unmasked.append(self.texts[text_str][sent_id_str]['token_list'])
        
            temp_context_labels_unmasked.append(self.texts[text_str][sent_id_str]['labels'])
        
            context_sent_ids_unmasked.append(sent_id_str)
        
            if sent_id_idx + i == sent_id_idx:
                focus_sentence_index = len(temp_context_unmasked) - 1
    
        temp_context['masked'], temp_context['unmasked'] = temp_context_masked, temp_context_unmasked
    
        temp_context_labels['masked'], temp_context_labels['unmasked'] = temp_context_labels_masked, \
        temp_context_labels_unmasked
    
        context_sent_ids['masked'], context_sent_ids['unmasked'] = context_sent_ids_masked, \
        context_sent_ids_unmasked
    
        return temp_context, temp_context_labels, focus_sentence_index, context_sent_ids
    
    def get_embeddings(self, folder_path, embed_dim = 300, embed_type = 'glove', first_time = False):
        
        if first_time:
            
            if embed_type == 'glove':
                
                embed_model = GloVe()
            
            self.embedding = torch.nn.Embedding(len(self.w2i), embed_dim)
            
            weight_matrix = np.zeros((len(self.w2i), embed_dim))
            
            weight_matrix[1, :] = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)
            
            weight_matrix[2:4, :] = np.random.randn(1, embed_dim)
            
            for word in self.w2i:
                
                if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS']:
                    
                    continue
                
                if word not in embed_model:
                    
                    weight_matrix[self.w2i[word], :] = weight_matrix[1, :]
                
                else:
                    
                    weight_matrix[self.w2i[word], :] = embed_model[word].numpy()
            
            self.embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
            
            with open(folder_path + 'vuamc_' + embed_type + '_vectors_gao.pckl', 'wb') as f:
                pickle.dump(self.embedding, f)
        else:
            with open(folder_path + 'vuamc_' + embed_type + '_vectors_gao.pckl', 'rb') as f:
                self.embedding = pickle.load(f)
    
    def get_elmo_embedding_dict(self, folder_path, first_time = False):
    
        if first_time:
            
            self.elmo_embed_dict = {}

            comb_set = pd.concat([self.original_dataset['train'], \
                                  self.original_dataset['val'], self.original_dataset['test']], \
                                 axis = 0, ignore_index = True)

            comb_set_rows = len(comb_set)

            batch_size = 64

            num_iterations = math.ceil(comb_set_rows / batch_size)

            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"

            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

            elmo = Elmo(options_file, weight_file, 1, dropout = 0)

            for x in range(num_iterations):
                
                start_index = x * batch_size

                end_index = min((x + 1) * batch_size, comb_set_rows)

                comb_set_slice = pd.DataFrame(comb_set[start_index : end_index])

                comb_set_slice_txt_sent_len = []

                comb_set_slice_sentences = []

                for index, row in comb_set_slice.iterrows():

                    tokenized_sent = row['sentence'].split()

                    comb_set_slice_txt_sent_len.append([row['txt_id'], str(row['sen_ix']), \
                                                        len(tokenized_sent), tokenized_sent])

                    comb_set_slice_sentences.append(tokenized_sent)

                comb_set_slice_char_ids = batch_to_ids(comb_set_slice_sentences)

                comb_set_slice_embeddings = elmo(comb_set_slice_char_ids)

                for j in range(len(comb_set_slice_txt_sent_len)):

                    text_id = comb_set_slice_txt_sent_len[j][0]

                    sent_id = comb_set_slice_txt_sent_len[j][1]

                    sent_id = self.text_sent_mapping[(text_id, sent_id)]

                    sent_len = comb_set_slice_txt_sent_len[j][2]

                    tokenized_sent = comb_set_slice_txt_sent_len[j][3]

                    for k in range(-1, sent_len):

                        if text_id not in self.elmo_embed_dict:
                            self.elmo_embed_dict[text_id] = {}

                        if sent_id not in self.elmo_embed_dict[text_id]:
                            self.elmo_embed_dict[text_id][sent_id] = {}

                        if k not in self.elmo_embed_dict[text_id][sent_id]:
                            self.elmo_embed_dict[text_id][sent_id][k] = {}

                        if k == -1:

                            glove_embedding = self.embedding(torch.LongTensor([0]))

                            elmo_embedding = torch.zeros(1, 1024)

                            embedding_concat = torch.cat((glove_embedding, elmo_embedding), dim = 1)

                            self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

                            self.elmo_embed_dict[text_id][sent_id][k]['word'] = '<PAD>'

                        else:

                            glove_embedding = self.embedding(torch.LongTensor([self.w2i[tokenized_sent[k]]]))

                            elmo_embedding = comb_set_slice_embeddings['elmo_representations'][0][j][k].view(1, 1024)

                            embedding_concat = torch.cat((glove_embedding, elmo_embedding), dim = 1)

                            self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

                            self.elmo_embed_dict[text_id][sent_id][k]['word'] = tokenized_sent[k]

            self.elmo_embed_dict["-1"] = {}
            self.elmo_embed_dict["-1"][-1] = {}
            self.elmo_embed_dict["-1"][-1][-1] = {}
            self.elmo_embed_dict["-1"][-1][-1]["glove_elmo"] = torch.zeros(1, 1324)
            self.elmo_embed_dict["-1"][-1][-1]["word"] = '<PAD>'

            with open(folder_path + 'vuamc_elmo_dict_gao.pckl', 'wb') as f:
                pickle.dump(self.elmo_embed_dict, f)

        else:

            with open(folder_path + "vuamc_elmo_dict_gao.pckl", 'rb') as f:
                self.elmo_embed_dict = pickle.load(f)

            for text_id in self.elmo_embed_dict.keys():
                
                for sent_id in self.elmo_embed_dict[text_id].keys():
                    
                    for word_id in self.elmo_embed_dict[text_id][sent_id].keys():
                        
                        self.elmo_embed_dict[text_id][sent_id][word_id]['glove_elmo'].requires_grad = False
                        
    def get_dataset_sent_ids(self):
        
        train_sent_ids, dev_sent_ids, test_sent_ids = [], [], []
        
        for dataset_type in ['train', 'val', 'test']:
            
            temp = []
            
            for index, row in self.original_dataset[dataset_type].iterrows():
                
                temp.append(self.text_sent_mapping[(row['txt_id'], str(row['sen_ix']))])
            
            if dataset_type == 'train':
                
                train_sent_ids.extend(temp)
            
            elif dataset_type == 'val':
                
                dev_sent_ids.extend(temp)
            
            elif dataset_type == 'test':
                
                test_sent_ids.extend(temp)
        
        return train_sent_ids, dev_sent_ids, test_sent_ids