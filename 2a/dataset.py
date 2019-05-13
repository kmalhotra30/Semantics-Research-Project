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
import ast
from allennlp.modules.elmo import Elmo, batch_to_ids
nltk.download('punkt')

# Defining the DatasetPreProcessor class
class DatasetPreProcessor:
  
  # Constructor Function
  def __init__(self, first_time = False):
    
    # Corpus name (For naming the files while saving)
    self.corpus_name = "vuamc"
    
    # Seed
    self.seed = 42
    
    # Retrieve vocabulary and information for all the texts 
    self.get_text_info()
    
    # Retrieve the word to index mapping
    self.get_word_to_index()
    
    # Retrieve the word embeddings
    self.get_embeddings(word_to_index = self.w2i, first_time = first_time)
    
  # Function to obtain the word to index mapping
  def get_word_to_index(self):
    
    self.w2i = {w : i + 4 for i, w in enumerate(list(self.vocab))}
    self.w2i['<PAD>'] = 0
    self.w2i['<UNK>'] = 1
    self.w2i['<SOS>'] = 2
    self.w2i['<EOS>'] = 3

  # Function to determine if a token is used metaphorically or not
  def is_metaphor(self, word):
    if len(word) > 2:
      if word[0] == 'M' and word[1] == '_':
        return 1
    return 0
    
  # Function to get the train, dev, and test splits
  def get_original_split_datasets(self, train_dev_split = 0.7):
    
    dataset_types = ['train', 'test']
    
    self.original_dataset = {}
    
    self.split_dataset = {}
    
    for dataset_type in dataset_types:
      
      self.original_dataset[dataset_type] = pd.read_csv('vuamc_corpus_' + dataset_type + '.csv', encoding = 'latin-1')
      
      self.original_dataset[dataset_type] = self.original_dataset[dataset_type].dropna()
     
    temp = shuffle(self.original_dataset['train'], random_state = self.seed)
    
    split_index = math.floor(len(temp) * train_dev_split)
    
    self.split_dataset['train'] = pd.DataFrame(temp[0 : split_index])
    
    self.split_dataset['dev'] = pd.DataFrame(temp[split_index : ])
    
    self.split_dataset['test'] = pd.DataFrame(self.original_dataset['test'])
    
    self.reset_index()
  
  # Function to reset the row numbering of a dataframe
  def reset_index(self):
    
    original_dataset_types = ['train', 'test']
    
    split_dataset_types = ['train', 'dev', 'test']
    
    for original_dataset_type in original_dataset_types:
      
      self.original_dataset[original_dataset_type].reset_index(inplace = True)
      
      self.original_dataset[original_dataset_type].drop("index", axis = 1, inplace = True)
    
    for split_dataset_type in split_dataset_types:
      
      self.split_dataset[split_dataset_type].reset_index(inplace = True)
      
      self.split_dataset[split_dataset_type].drop("index", axis = 1, inplace = True)

  # Function to get the information pertaining to all the texts present in the corpus
  def get_text_info(self):
    
    self.vocab = set()
    
    self.texts = {}
    
    self.text_id_mapping = {}
    
    train_corpus = pd.read_csv('vuamc_corpus_train.csv', encoding = 'latin-1')
    
    test_corpus = pd.read_csv('vuamc_corpus_test.csv', encoding = 'latin-1')
    
    train_corpus = train_corpus.dropna()
    
    test_corpus = test_corpus.dropna()
    
    comb = pd.concat([train_corpus, test_corpus], axis = 0, ignore_index = True)
    
    self.text_id_mapping["-1"] = -1
    
    counter = 0
      
    for index, row in comb.iterrows():

      if row['txt_id'] not in self.texts:

        self.texts[row['txt_id']] = {}
        
      if row['txt_id'] not in self.text_id_mapping:
        
        self.text_id_mapping[row['txt_id']] = counter
        counter += 1

      sent_id = row['sentence_id']
      
      if isinstance(sent_id, str):
        sent_id = int(re.sub("\D", "", sent_id))

      if sent_id not in self.texts[row['txt_id']]:

        self.texts[row['txt_id']][sent_id] = {}
      
      self.texts[row['txt_id']][sent_id]['token_list'] = nltk.word_tokenize(row['sentence_txt'])
      
      self.vocab = self.vocab.union(set(self.texts[row['txt_id']][sent_id]['token_list']))
      
      self.texts[row['txt_id']][sent_id]['labels'] = list(map(self.is_metaphor, self.texts[row['txt_id']][sent_id]['token_list']))
   
    self.text_id_mapping_reverse = {v : k for k, v in self.text_id_mapping.items()}
      
  # Function to get the word embeddings
  def get_embeddings(self, word_to_index, embed_dim = 300, embed_type = 'glove', first_time = False):

    if first_time:
      
      if embed_type == 'glove':
        embed_model = GloVe()
      
      self.embedding = torch.nn.Embedding(len(word_to_index), embed_dim)

      weight_matrix = np.zeros((len(word_to_index), embed_dim))
      
      weight_matrix[1, :] = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)

      weight_matrix[2:4, :] = np.random.randn(1, embed_dim)

      for word in word_to_index:
        
        if word in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
          continue
          
        if word not in embed_model:
            weight_matrix[word_to_index[word], :] = weight_matrix[1, :]
        
        else:
            weight_matrix[word_to_index[word], :] = embed_model[word].numpy()
      
      self.embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
      
      with open(self.corpus_name + "_" + embed_type + "_vectors.pckl", 'wb') as f:
        pickle.dump(self.embedding, f)
    
    else:
      
      with open(self.corpus_name + "_" + embed_type + "_vectors.pckl", 'rb') as f:
          self.embedding = pickle.load(f)

  # Function to get the context around a focus sentence   
  def get_context(self, text_id, sent_id, context_window):
    
    if isinstance(sent_id, str):
        sent_id = int(re.sub("\D", "", sent_id))
    
    temp_context = {}
    
    temp_context_labels = {}
    
    context_sent_ids = {}
    
    temp_context_masked = []

    temp_context_unmasked = []
    
    temp_context_labels_masked = []
    
    temp_context_labels_unmasked = []
    
    context_sent_ids_masked = []
    
    context_sent_ids_unmasked = []
    
    sorted_sent_ids = sorted(list(self.texts[text_id]))
    
    sent_id_idx = sorted_sent_ids.index(sent_id)
    
    for i in range(-context_window, context_window + 1):
      
      if sent_id_idx + i < 0 or sent_id_idx + i >= len(sorted_sent_ids):
        
        pass
        
      else:
        
        if sent_id_idx + i != sent_id_idx:
          temp_context_masked.append(self.texts[text_id][sorted_sent_ids[sent_id_idx + i]]['token_list'])
          temp_context_labels_masked.append(self.texts[text_id][sorted_sent_ids[sent_id_idx + i]]['labels'])
          context_sent_ids_masked.append(sorted_sent_ids[sent_id_idx + i])
        
        temp_context_unmasked.append(self.texts[text_id][sorted_sent_ids[sent_id_idx + i]]['token_list'])
        
        temp_context_labels_unmasked.append(self.texts[text_id][sorted_sent_ids[sent_id_idx + i]]['labels'])
        
        context_sent_ids_unmasked.append(sorted_sent_ids[sent_id_idx + i])
        
        if sent_id_idx + i == sent_id_idx:
          focus_sentence_index = len(temp_context_unmasked) - 1
    
    temp_context['masked'], temp_context['unmasked'] = temp_context_masked, temp_context_unmasked
    
    temp_context_labels['masked'], temp_context_labels['unmasked'] = temp_context_labels_masked, temp_context_labels_unmasked
    
    context_sent_ids['masked'], context_sent_ids['unmasked'] = context_sent_ids_masked, context_sent_ids_unmasked
    
    return temp_context, temp_context_labels, focus_sentence_index, context_sent_ids
  
  # Function to get the elmo embedding dictionary
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

          tokenized_sent = nltk.word_tokenize(row['sentence_txt'])

          comb_set_slice_txt_sent_len.append([row['txt_id'], row['sentence_id'], len(tokenized_sent), tokenized_sent])

          comb_set_slice_sentences.append(tokenized_sent)

        comb_set_slice_char_ids = batch_to_ids(comb_set_slice_sentences)

        comb_set_slice_embeddings = elmo(comb_set_slice_char_ids)

        for j in range(len(comb_set_slice_txt_sent_len)):

          text_id = comb_set_slice_txt_sent_len[j][0]

          sent_id = comb_set_slice_txt_sent_len[j][1]

          if isinstance(sent_id, str):
            sent_id = int(re.sub("\D", "", sent_id))

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

              self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = list(torch.cat((glove_embedding, elmo_embedding), dim = 1).detach().numpy().squeeze()) # (1, 1324)

              #self.elmo_embed_dict[text_id][sent_id][k]['elmo'] = elmo_embedding # (1, 1024)

              #self.elmo_embed_dict[text_id][sent_id][k]['glove'] = glove_embedding # (1, 300)

              self.elmo_embed_dict[text_id][sent_id][k]['word'] = '<PAD>'

            else:

              glove_embedding = self.embedding(torch.LongTensor([self.w2i[tokenized_sent[k]]]))

              elmo_embedding = comb_set_slice_embeddings['elmo_representations'][0][j][k].view(1, 1024)

              self.elmo_embed_dict[text_id][sent_id][k]['glove_elmo'] = list(torch.cat((glove_embedding, elmo_embedding), dim = 1).detach().numpy().squeeze()) # (1, 1324)

              #self.elmo_embed_dict[text_id][sent_id][k]['elmo'] = elmo_embedding # (1, 1024)

              #self.elmo_embed_dict[text_id][sent_id][k]['glove'] = glove_embedding # (1, 300)

              self.elmo_embed_dict[text_id][sent_id][k]['word'] = tokenized_sent[k]

      self.elmo_embed_dict["-1"] = {}
      self.elmo_embed_dict["-1"][-1] = {}
      self.elmo_embed_dict["-1"][-1][-1] = {}
      self.elmo_embed_dict["-1"][-1][-1]["glove_elmo"] = list(torch.zeros(1, 1324).detach().numpy().squeeze())
      #self.elmo_embed_dict["-1"][-1][-1]["elmo"] = torch.zeros(1, 1024)
      #self.elmo_embed_dict["-1"][-1][-1]["glove"] = torch.zeros(1, 300)
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