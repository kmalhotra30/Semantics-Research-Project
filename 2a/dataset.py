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

class DatasetPreProcessor:
  
  def __init__(self, seed = 42):
    
    self.corpus_name = 'vuamc'
    
    self.seed = seed
    
  def pre_process_corpus(self):
    
    self.texts = {}
    
    self.vocab = set()
    
    for dataset_type in ['train', 'test']:
    
      train_corpus = pd.read_csv('vuamc_corpus_' + dataset_type + '.csv', encoding = 'utf-8')

      train_corpus = train_corpus.dropna()

      train_corpus_all_pos_tokens = pd.read_csv('all_pos_tokens_' + dataset_type + '.csv', header = None, encoding = 'utf-8')

      for index, row in train_corpus.iterrows():

        text_id = row['txt_id']

        sent_id = str(row['sentence_id'])

        tokenized_sentence = list(map(self.pre_process_word, nltk.word_tokenize(row['sentence_txt'])))
        
        self.vocab = self.vocab.union(set(tokenized_sentence))

        if text_id not in self.texts:

          self.texts[text_id] = {}

        if sent_id not in self.texts[text_id]:

          self.texts[text_id][sent_id] = {}

        self.texts[text_id][sent_id]['token_list'] = tokenized_sentence

        self.texts[text_id][sent_id]['labels'] = [0] * len(tokenized_sentence)

      for index, row in train_corpus_all_pos_tokens.iterrows():

        token_id = row[0]

        text_id = token_id.split('_')[0]

        sent_id = token_id.split('_')[1]

        word_id = int(token_id.split('_')[2]) - 1

        self.texts[text_id][sent_id]['labels'][word_id] = int(row[1])
   
  def get_mappings(self):
    
    self.text_id_mapping = {}
    
    self.text_sent_mapping = {}
    
    text_counter = 0
    
    text_sent_counter = 0
    
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
    
  def get_word_to_index(self):
    
    self.w2i = {w : i + 4 for i, w in enumerate(list(self.vocab))}
    self.w2i['<PAD>'] = 0
    self.w2i['<UNK>'] = 1
    self.w2i['<SOS>'] = 2
    self.w2i['<EOS>'] = 3
    
  def pre_process_word(self, word):
    
    if len(word) > 2:
      
      if word[0] in ['M', 'm'] and word[1] == '_':
        
        return word[2 : ]
    
    return word     
  
  def get_context(self, text_id, sent_id, context_window):
    
    temp_context, temp_context_labels, context_sent_ids = {}, {}, {}
    
    temp_context_masked, temp_context_unmasked = [], []
    
    temp_context_labels_masked, temp_context_labels_unmasked = [], []
    
    context_sent_ids_masked, context_sent_ids_unmasked = [], []
    
    (text_str, sent_str) = self.text_sent_mapping_reverse[sent_id]
    
    sorted_sent_ids = sorted(list(self.texts[text_str]))
    
    sent_id_idx = sorted_sent_ids.index(sent_str)
    
    for i in range(-context_window, context_window + 1):
      
      if sent_id_idx + i < 0 or sent_id_idx + i >= len(sorted_sent_ids):
        
        pass
        
      else:
        
        if sent_id_idx + i != sent_id_idx:
          temp_context_masked.append(self.texts[text_str][sorted_sent_ids[sent_id_idx + i]]['token_list'])
          temp_context_labels_masked.append(self.texts[text_str][sorted_sent_ids[sent_id_idx + i]]['labels'])
          context_sent_ids_masked.append(sorted_sent_ids[sent_id_idx + i])
        
        temp_context_unmasked.append(self.texts[text_str][sorted_sent_ids[sent_id_idx + i]]['token_list'])
        
        temp_context_labels_unmasked.append(self.texts[text_str][sorted_sent_ids[sent_id_idx + i]]['labels'])
        
        context_sent_ids_unmasked.append(sorted_sent_ids[sent_id_idx + i])
        
        if sent_id_idx + i == sent_id_idx:
          focus_sentence_index = len(temp_context_unmasked) - 1
    
    temp_context['masked'], temp_context['unmasked'] = temp_context_masked, temp_context_unmasked
    
    temp_context_labels['masked'], temp_context_labels['unmasked'] = temp_context_labels_masked, temp_context_labels_unmasked
    
    context_sent_ids['masked'], context_sent_ids['unmasked'] = context_sent_ids_masked, context_sent_ids_unmasked
    
    return temp_context, temp_context_labels, focus_sentence_index, context_sent_ids
  
  def get_elmo_embedding_dict(self, folder_path, first_time = False):
    
    if first_time:
      
      self.elmo_embed_dict = {}

      comb_set = pd.concat([self.original_dataset['train'], self.original_dataset['test']], axis = 0, ignore_index = True)

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
          
          tokenized_sent = list(map(self.pre_process_word, nltk.word_tokenize(row['sentence_txt'])))

          comb_set_slice_txt_sent_len.append([row['txt_id'], str(row['sentence_id']), len(tokenized_sent), tokenized_sent])

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
              
              embedding_concat.requires_grad = False

              self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

              self.elmo_embed_dict[text_id][sent_id][k]['word'] = '<PAD>'

            else:

              glove_embedding = self.embedding(torch.LongTensor([self.w2i[tokenized_sent[k]]]))

              elmo_embedding = comb_set_slice_embeddings['elmo_representations'][0][j][k].view(1, 1024)
              
              embedding_concat = torch.cat((glove_embedding, elmo_embedding), dim = 1)
              
              embedding_concat.requires_grad = False

              self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

              self.elmo_embed_dict[text_id][sent_id][k]['word'] = tokenized_sent[k]

      self.elmo_embed_dict["-1"] = {}
      self.elmo_embed_dict["-1"][-1] = {}
      self.elmo_embed_dict["-1"][-1][-1] = {}
      self.elmo_embed_dict["-1"][-1][-1]["glove_elmo"] = torch.zeros(1, 1324)
      self.elmo_embed_dict["-1"][-1][-1]["word"] = '<PAD>'
      
      with open(folder_path + self.corpus_name + "_elmo_dict.pckl", 'wb') as f:
         pickle.dump(self.elmo_embed_dict, f)
        
    else:
      
      with open(folder_path + self.corpus_name + "_elmo_dict.pckl", 'rb') as f:
          self.elmo_embed_dict = pickle.load(f)
       
      for text_id in self.elmo_embed_dict.keys():
        
        for sent_id in self.elmo_embed_dict[text_id].keys():
          
          for word_id in self.elmo_embed_dict[text_id][sent_id].keys():
            
            self.elmo_embed_dict[text_id][sent_id][word_id]['glove_elmo'].requires_grad = False
   
  
  def get_elmo_embedding_dict_wider_context(self, folder_path, context_window = 2, first_time = False):
    
    if first_time:
      
      self.elmo_embed_dict_wider_context = {}

      comb_set = pd.concat([self.original_dataset['train'], self.original_dataset['test']], axis = 0, ignore_index = True)

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
          
          context, _, focus_sentence_index, _ = self.get_context(row['txt_id'], self.text_sent_mapping[(row['txt_id'], str(row['sentence_id']))], context_window = context_window)
          
          tokenized_sent_len = []
          
          tokenized_sent = []
          
          for context_sentence in context['unmasked']:
            
            tokenized_sent_len.append(len(context_sentence))
            
            tokenized_sent.extend(context_sentence)
          
          offset = 0
          
          for i in range(len(tokenized_sent_len)):
            
            if i == focus_sentence_index:
              break
              
            offset += tokenized_sent_len[i]
            
          comb_set_slice_txt_sent_len.append([row['txt_id'], str(row['sentence_id']), len(context['unmasked'][focus_sentence_index]), context['unmasked'][focus_sentence_index], offset])

          comb_set_slice_sentences.append(tokenized_sent)

        comb_set_slice_char_ids = batch_to_ids(comb_set_slice_sentences)

        comb_set_slice_embeddings = elmo(comb_set_slice_char_ids)

        for j in range(len(comb_set_slice_txt_sent_len)):

          text_id = comb_set_slice_txt_sent_len[j][0]

          sent_id = comb_set_slice_txt_sent_len[j][1]
          
          sent_id = self.text_sent_mapping[(text_id, sent_id)]

          sent_len = comb_set_slice_txt_sent_len[j][2]

          tokenized_sent = comb_set_slice_txt_sent_len[j][3]
          
          offset = comb_set_slice_txt_sent_len[j][4]

          for k in range(-1, sent_len):

            if text_id not in self.elmo_embed_dict_wider_context:
              self.elmo_embed_dict_wider_context[text_id] = {}
            
            if sent_id not in self.elmo_embed_dict_wider_context[text_id]:
              self.elmo_embed_dict_wider_context[text_id][sent_id] = {}

            if k not in self.elmo_embed_dict_wider_context[text_id][sent_id]:
              self.elmo_embed_dict_wider_context[text_id][sent_id][k] = {}

            if k == -1:

              glove_embedding = self.embedding(torch.LongTensor([0]))

              elmo_embedding = torch.zeros(1, 1024)
              
              embedding_concat = torch.cat((glove_embedding, elmo_embedding), dim = 1)

              self.elmo_embed_dict_wider_context[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

              self.elmo_embed_dict_wider_context[text_id][sent_id][k]['word'] = '<PAD>'

            else:

              glove_embedding = self.embedding(torch.LongTensor([self.w2i[tokenized_sent[k]]]))

              elmo_embedding = comb_set_slice_embeddings['elmo_representations'][0][j][offset + k].view(1, 1024)
              
              embedding_concat = torch.cat((glove_embedding, elmo_embedding), dim = 1)

              self.elmo_embed_dict_wider_context[text_id][sent_id][k]['glove_elmo'] = embedding_concat # (1, 1324)

              self.elmo_embed_dict_wider_context[text_id][sent_id][k]['word'] = tokenized_sent[k]

      self.elmo_embed_dict_wider_context["-1"] = {}
      self.elmo_embed_dict_wider_context["-1"][-1] = {}
      self.elmo_embed_dict_wider_context["-1"][-1][-1] = {}
      self.elmo_embed_dict_wider_context["-1"][-1][-1]["glove_elmo"] = torch.zeros(1, 1324)
      self.elmo_embed_dict_wider_context["-1"][-1][-1]["word"] = '<PAD>'
      
      with open(folder_path + self.corpus_name + "_elmo_dict_wider_context.pckl", 'wb') as f:
         pickle.dump(self.elmo_embed_dict_wider_context, f)
        
    else:
      
      with open(folder_path + self.corpus_name + "_elmo_dict_wider_context.pckl", 'rb') as f:
          self.elmo_embed_dict_wider_context = pickle.load(f)
       
      for text_id in self.elmo_embed_dict_wider_context.keys():
        
        for sent_id in self.elmo_embed_dict_wider_context[text_id].keys():
          
          for word_id in self.elmo_embed_dict_wider_context[text_id][sent_id].keys():
            
            self.elmo_embed_dict_wider_context[text_id][sent_id][word_id]['glove_elmo'].requires_grad = False
            
  def get_embeddings(self, folder_path, embed_dim = 300, embed_type = 'glove', first_time = False):

    if first_time:
      
      if embed_type == 'glove':
        embed_model = GloVe()
      
      self.embedding = torch.nn.Embedding(len(self.w2i), embed_dim)

      weight_matrix = np.zeros((len(self.w2i), embed_dim))
      
      weight_matrix[1, :] = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)

      weight_matrix[2:4, :] = np.random.randn(1, embed_dim)

      for word in self.w2i:
        
        if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
          continue
          
        if word not in embed_model:
            weight_matrix[self.w2i[word], :] = weight_matrix[1, :]
        
        else:
            weight_matrix[self.w2i[word], :] = embed_model[word].numpy()
      
      self.embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
      
      with open(folder_path + self.corpus_name + "_" + embed_type + "_vectors.pckl", 'wb') as f:
        pickle.dump(self.embedding, f)
    
    else:
      
      with open(folder_path + self.corpus_name + "_" + embed_type + "_vectors.pckl", 'rb') as f:
          self.embedding = pickle.load(f)
          
  def get_original_split_datasets(self, folder_path, train_dev_split = 0.9, first_time = False):
    
    dataset_types = ['train', 'test']
    
    self.original_dataset = {}
    
    self.split_dataset = {}
    
    for dataset_type in dataset_types:
      
      self.original_dataset[dataset_type] = pd.read_csv('vuamc_corpus_' + dataset_type + '.csv', encoding = 'utf-8')
      
      self.original_dataset[dataset_type] = self.original_dataset[dataset_type].dropna()
      
    if first_time:
     
      temp = self.get_dataframe(self.original_dataset['train'])
      
      temp = shuffle(temp, random_state = self.seed)
      
      temp.reset_index(inplace = True)
      
      temp.drop("index", axis = 1, inplace = True)

      split_index = math.floor(len(temp) * train_dev_split)

      self.split_dataset['train'] = temp[0 : split_index]

      self.split_dataset['dev'] = temp[split_index : ]

      self.split_dataset['test'] = self.get_dataframe(self.original_dataset['test'])
      
      self.reset_index()
      
      with open(folder_path + 'train_set.pckl', 'wb') as f:
        pickle.dump(self.split_dataset['train'], f)

      with open(folder_path + 'dev_set.pckl', 'wb') as f:
        pickle.dump(self.split_dataset['dev'], f)
  
      with open(folder_path + 'test_set.pckl', 'wb') as f:
        pickle.dump(self.split_dataset['test'], f)
    
    else:
      
      with open(folder_path + 'train_set.pckl', 'rb') as f:
        self.split_dataset['train'] = pickle.load(f)

      with open(folder_path + 'dev_set.pckl', 'rb') as f:
        self.split_dataset['dev'] = pickle.load(f)
  
      with open(folder_path + 'test_set.pckl', 'rb') as f:
        self.split_dataset['test'] = pickle.load(f)
      
  def get_dataframe(self, source_df):
    
    target_df = pd.DataFrame(columns = ['txt_id', 'sentence_id', 'word_id'])
    
    c = 0
    
    l = 0
    
    for index, row in source_df.iterrows():
      
      text_id = row['txt_id']
      
      sent_id = str(row['sentence_id'])
      
      tokenized_sent = list(map(self.pre_process_word, nltk.word_tokenize(row['sentence_txt'])))
      
      for i in range(len(tokenized_sent)):
        
        target_df.loc[c] = [self.text_id_mapping[text_id], self.text_sent_mapping[(text_id, sent_id)], i]
        
        c += 1
      
      l += 1
      
      print(l)
    
    return target_df
    
  def reset_index(self):
    
    original_dataset_types = ['train', 'test']
    
    split_dataset_types = ['train', 'dev', 'test']
    
    for original_dataset_type in original_dataset_types:
      
      self.original_dataset[original_dataset_type].reset_index(inplace = True)
      
      self.original_dataset[original_dataset_type].drop("index", axis = 1, inplace = True)
    
    for split_dataset_type in split_dataset_types:
      
      self.split_dataset[split_dataset_type].reset_index(inplace = True)
      
      self.split_dataset[split_dataset_type].drop("index", axis = 1, inplace = True)