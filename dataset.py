# Importing the Libraries
import nltk
import numpy as np
import torch
import math
import torchnlp
from torchnlp.word_to_vector import GloVe
import pickle
nltk.download('semcor')
nltk.download('wordnet')
nltk.download('punkt')

# DatasetPreProcessor class
class DatasetPreProcessor:
  
  ################ Constructor Function ########################################
  def __init__(self, corpus = 'semcor'):
    
    # Initializing the corpus
    self.corpus = nltk.corpus.semcor if corpus == 'semcor' else nltk.corpus.masc
    
    # Determining the number of documents in the corpus
    self.num_docs = len(self.corpus.fileids())
  
  ################ Index to Sentence Mapping ###################################
  def get_index_2_sentence(self, doc_id):
    
    return {i : s for i, s in enumerate(self.corpus.sents(doc_id))}
  
  ################ Vocabulary, word2index and index2word mapping ###############
  def get_vocab(self, doc_ids):
    
    # Vocabulary
    vocab = set(self.corpus.words(doc_ids))
    
    # Word to index mapping
    w2i = {w : i for i, w in enumerate(list(vocab))}
    
    # Index to word mapping
    i2w = {i : w for i, w in enumerate(list(vocab))}
    
    return vocab, w2i, i2w
  
  ################ Fetch all the sentences of a document given its document ID #######################
  def get_document(self, doc_id):
    
    return self.corpus.sents(doc_id)
  
  ################ Fetch all the words of a document given its document ID ###########################
  def get_words(self, doc_id):
    
    return self.corpus.words(doc_id)
  
  ################ Fetch the words that need to be disambiguated from a document given its document ID #################
  def get_wsd_words(self, doc_id):
    
    # Dictionary to store the words that need to be disambiguated
    wsd = {}
 
    # To store the sentence index
    counter = 0
  
    # Iterating through the tagged sentences in a document
    for tagged_sentence in self.corpus.tagged_sents(doc_id, tag = 'sem'):
      
      # For each tagged chunk of the tagged sentence
      for tagged_chunk in tagged_sentence:
      
        # If the tagged chunk is a tree
        if type(tagged_chunk) == nltk.tree.Tree:

          # If the value of the tree is not a subtree and is a list of size 1
          if type(tagged_chunk[0]) != nltk.tree.Tree and len(tagged_chunk) == 1:

            # Add to the dictionary if not present
            if tagged_chunk[0] not in wsd:
              wsd[tagged_chunk[0]] = {}

            # If the tree label is a Lemma
            if type(tagged_chunk.label()) == nltk.corpus.reader.wordnet.Lemma:
              
              # If the synset is not present in the dictionary
              if tagged_chunk.label().synset() not in wsd[tagged_chunk[0]]:
                
                wsd[tagged_chunk[0]][tagged_chunk.label().synset()] = []
    
              wsd[tagged_chunk[0]][tagged_chunk.label().synset()].append(counter)
      
      # Increment sentence index
      counter += 1
      
    return wsd
  
  ###### To obtain the context sentences given the document ID and sentence ID in which a particular word occurs ###############################
  def get_context_sentences(self, doc_id, sent_id, context_num):
    
    # Temporary list to store the context sentences
    temp_context = []
    
    # Defining the window of context
    for i in range(-context_num, context_num + 1):
      
      # To handle the corner cases
      if sent_id + i < 0 or sent_id + i >= len(self.corpus.sents(doc_id)):
        
        temp_context.append('<PAD>')
    
      else:
        
        temp_context.append(self.document[doc_id]['i2s'][sent_id + i])
    
    return temp_context
  
  ######## To obtain the context words given the document ID in which a particular word occurs #########################
  def get_context_words(self, doc_id, word, context_num):
    
    # Temporary list to store the context words
    word_context = []
    
    # Obtaining the list of words in the document
    doc_words = self.corpus.words(doc_id)
    
    # Obtaining the occurrences of a particular word in the document
    word_occurrences = [i for i, w in enumerate(doc_words) if w == word]
    
    # Obtaining the context window for each occurrence
    for occurrence in word_occurrences:
      
      temp_context = []
      
      for i in range(-context_num, context_num + 1):
        
        if occurrence + i < 0 or occurrence + i >= len(doc_words):
        
          temp_context.append('<PAD>')
        
        else:
        
          temp_context.append(doc_words[occurrence + i])
      
      word_context.append(temp_context)
    
    return word_context

  ######################## Get word, sentence or document-level contexts for a given word and its sense ##############################
  def get_wsd_context(self, dataset_type, word, word_sense, context_num = 0, is_word = False, is_sentence = True, is_document = False):
    
    # Document Level Context
    if is_document == True:
      
      context = [self.get_document(doc_id) for doc_id in self.doc_ids[dataset_type] if word in self.document[doc_id]['wsd'] and \
                 word_sense in self.document[doc_id]['wsd'][word]]
    
    # Sentence Level Context
    elif is_sentence == True:
      
      context = []
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            sentence_ids = self.document[doc_id]['wsd'][word][word_sense]
            
            for sent_id in sentence_ids:
              context.append(self.get_context_sentences(doc_id, sent_id, context_num))
    
    # Word Level Context
    elif is_word == True:
      
      context = []
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            context.append(self.get_context_words(doc_id, word, context_num))
            
    return context
  
  # To get the word embeddings (Currently glove)
  def get_embeddings(self, word_to_index, embed_dim = 300, embed_type = 'glove'):

    if embed_type == 'glove':
      embed_model = GloVe()
    else:
      pass

    embedding = torch.nn.Embedding(len(word_to_index) + 4, embed_dim)

    weight_matrix = np.zeros((len(word_to_index) + 4, embed_dim))
    
    # Initializing vectors for the <UNK>, <SOS> and <EOS> tokens
    weight_matrix[1:4, :] = np.random.randn(1, embed_dim)
    
    for word in word_to_index:
      weight_matrix[word_to_index[word] + 4, :] = embed_model[word].numpy()

    embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
  
    return embedding

  # Get the train and validation datasets given the split
  def get_train_validation_split(self, split):
    
    self.doc_ids, self.vocab, self.i2w, self.w2i = {}, {}, {}, {}
    
    split_index = math.floor(self.num_docs * split)
    
    self.doc_ids['train'] = self.corpus.fileids()[0 : split_index]
    
    self.doc_ids['dev'] = self.corpus.fileids()[split_index : ]
    
    self.vocab['train'], self.w2i['train'], self.i2w['train'] = self.get_vocab(self.doc_ids['train'])
    
    self.vocab['dev'], self.w2i['dev'], self.i2w['dev'] = self.get_vocab(self.doc_ids['dev'])
    
    self.preprocess_corpus()
    
  # Obtain the information about each document
  def preprocess_corpus(self):
    
    self.document = {}
        
    for k in self.corpus.fileids():
      
      self.document[k] = {}
      
      self.document[k]['i2s'] = self.get_index_2_sentence(k)
      
      self.document[k]['wsd'] = self.get_wsd_words(k)