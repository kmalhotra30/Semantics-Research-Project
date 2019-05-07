# Importing Libraries
import nltk
import numpy as np
import torch
import math
import torchnlp
from torchnlp.word_to_vector import GloVe
import pickle
import random
nltk.download('semcor')
nltk.download('wordnet')
nltk.download('punkt')

class DatasetPreProcessor:
  
  def __init__(self, corpus = 'semcor', first_time = False):
    
    self.corpus_name = corpus
    
    self.corpus = nltk.corpus.semcor if self.corpus_name == 'semcor' else nltk.corpus.masc
    
    self.wordnet = nltk.corpus.wordnet
    
    self.num_docs = len(self.corpus.fileids())
    
    self.get_vocab()
    
    if first_time:
      self.document = self.preprocess_corpus()
      with open("document_" + corpus + ".pckl", 'wb') as f:
        pickle.dump(self.document, f)
    else:
      with open("document_" + corpus + ".pckl", 'rb') as f:
        self.document = pickle.load(f)
        
    self.get_embeddings(self.w2i, first_time = first_time)
  
  def get_index_2_sentence(self, doc_id):
    
    return {i : s for i, s in enumerate(self.corpus.sents(doc_id))}
  
  def get_vocab(self):
    
    self.vocab = set(self.corpus.words())
    
    self.w2i = {w : i + 4 for i, w in enumerate(list(self.vocab))}
    self.w2i['<PAD>'] = 0
    self.w2i['<UNK>'] = 1
    self.w2i['<SOS>'] = 2
    self.w2i['<EOS>'] = 3

    self.i2w = {i + 4 : w for i, w in enumerate(list(self.vocab))}
    self.i2w[0] = '<PAD>'
    self.i2w[1] = '<UNK>'
    self.i2w[2] = '<SOS>'
    self.i2w[3] = '<EOS>'
  
  def get_document(self, doc_id):
    
    return self.corpus.sents(doc_id)
  
  def get_words(self, doc_id):
    
    return self.corpus.words(doc_id)
  
  def get_definition(self, synset_name):
    
    print(self.wordnet.synset(synset_name).definition())
  
  def get_wsd_words(self, doc_id):
    
    wsd = {}
 
    counter = 0
  
    for tagged_sentence in self.corpus.tagged_sents(doc_id, tag = 'sem'):
      
      for tagged_chunk in tagged_sentence:
      
        if type(tagged_chunk) == nltk.tree.Tree:

          if type(tagged_chunk[0]) != nltk.tree.Tree and len(tagged_chunk) == 1:

            if tagged_chunk[0] not in wsd:
              wsd[tagged_chunk[0]] = {}

            if type(tagged_chunk.label()) == nltk.corpus.reader.wordnet.Lemma:
              
              if tagged_chunk.label().synset().name() not in wsd[tagged_chunk[0]]:
                
                wsd[tagged_chunk[0]][tagged_chunk.label().synset().name()] = []
                
              wsd[tagged_chunk[0]][tagged_chunk.label().synset().name()].append(counter)
      
      counter += 1
      
    return wsd
  
  def get_context_sentences(self, doc_id, sent_id, context_num):
    
    temp_context_masked = []

    temp_context_unmasked = []
    
    for i in range(-context_num, context_num + 1):
      
      if sent_id + i < 0 or sent_id + i >= len(self.corpus.sents(doc_id)):
        
        temp_context_masked.append('<PAD>')

        temp_context_unmasked.append('<PAD>')
        
      else:
        
        if sent_id + i != sent_id:
          temp_context_masked.append(self.document[doc_id]['i2s'][sent_id + i])
        
        temp_context_unmasked.append(self.document[doc_id]['i2s'][sent_id + i])       
    
    return temp_context_masked, temp_context_unmasked
  
  def get_context_words(self, doc_id, word, context_num):
    
    word_context_masked = []

    word_context_unmasked = []
    
    doc_words = self.corpus.words(doc_id)
    
    word_occurrences = [i for i, w in enumerate(doc_words) if w == word]
    
    for occurrence in word_occurrences:
      
      temp_context_masked = []

      temp_context_unmasked = []
      
      for i in range(-context_num, context_num + 1):
        
        if occurrence + i < 0 or occurrence + i >= len(doc_words):
        
          temp_context_masked.append('<PAD>')
          
          temp_context_unmasked.append('<PAD>')
        
        else:
        
          if occurrence + i != occurrence:
            temp_context_masked.append(doc_words[occurrence + i])
          
          temp_context_unmasked.append(doc_words[occurrence + i])
      
      word_context_masked.append(temp_context_masked)

      word_context_unmasked.append(temp_context_unmasked)
    
    return word_context_masked, word_context_unmasked
  
  def get_wsd_senses(self, word, doc_id_list):

    word_senses = []

    for doc_id in doc_id_list:

      if word in self.document[doc_id]['wsd']:
        word_sense_list = list(self.document[doc_id]['wsd'][word])

        word_senses.extend(word_sense_list)
    
    return list(set(word_senses))

  def get_wsd_context(self, dataset_type, word, word_sense, context_num = 0, is_word = False, is_sentence = True, is_document = False):
    
    if is_document == True:
      
      context = [self.get_document(doc_id) for doc_id in self.doc_ids[dataset_type] if word in self.document[doc_id]['wsd'] and \
                 word_sense in self.document[doc_id]['wsd'][word]]
    
    elif is_sentence == True:
      
      context = {"masked" : [], "unmasked" : []}
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            sentence_ids = self.document[doc_id]['wsd'][word][word_sense]
            
            for sent_id in sentence_ids:
              masked, unmasked = self.get_context_sentences(doc_id, sent_id, context_num)

              context["masked"].append(masked)

              context["unmasked"].append(unmasked)
    
    elif is_word == True:
      
      context = {"masked" : [], "unmasked" : []}
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            masked, unmasked = self.get_context_words(doc_id, word, context_num)

            context["masked"].append(masked)

            context["unmasked"].append(unmasked)
            
    return context
  
  def get_embeddings(self, word_to_index, embed_dim = 300, embed_type = 'glove', first_time = False):

    if first_time:
      
      if embed_type == 'glove':
        embed_model = GloVe()
      
      self.embedding = torch.nn.Embedding(len(word_to_index), embed_dim)

      weight_matrix = np.zeros((len(word_to_index), embed_dim))

      weight_matrix[1, :] = np.random.uniform(-0.05, 0.05, 300).astype(np.float32)

      weight_matrix[2:4, :] = np.random.randn(1, embed_dim)

      for word in word_to_index:
        
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
    
  def get_train_validation_split(self, split):
    
    self.doc_ids = {}
    
    split_index = math.floor(self.num_docs * split)

    doc_id_list = list(self.corpus.fileids())
    
    random.shuffle(doc_id_list, random.random)

    self.doc_ids['train'] = doc_id_list[0 : split_index]
    
    self.doc_ids['dev'] = doc_id_list[split_index : ]
    
  def preprocess_corpus(self):
    
    document = {}
        
    for k in self.corpus.fileids():
      
      document[k] = {}
      
      document[k]['i2s'] = self.get_index_2_sentence(k)
      
      document[k]['wsd'] = self.get_wsd_words(k)
      
    return document