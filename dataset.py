class DatasetPreProcessor:
  
  def __init__(self, corpus = 'semcor', first_time = False):
    
    self.corpus = nltk.corpus.semcor if corpus == 'semcor' else nltk.corpus.masc
    
    self.wordnet = nltk.corpus.wordnet
    
    self.num_docs = len(self.corpus.fileids())
    
    if first_time:
      self.document = self.preprocess_corpus()
      with open("document_" + corpus + ".pickle", 'wb') as f:
        pickle.dump(self.document, f)
    else:
      with open("document_" + corpus + ".pickle", 'rb') as f:
        self.document = pickle.load(f)
  
  def get_index_2_sentence(self, doc_id):
    
    return {i : s for i, s in enumerate(self.corpus.sents(doc_id))}
  
  def get_vocab(self, doc_ids):
    
    vocab = set(self.corpus.words(doc_ids))
    
    w2i = {w : i for i, w in enumerate(list(vocab))}
    
    i2w = {i : w for i, w in enumerate(list(vocab))}
    
    return vocab, w2i, i2w
  
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
    
    temp_context = []
    
    for i in range(-context_num, context_num + 1):
      
      if sent_id + i < 0 or sent_id + i >= len(self.corpus.sents(doc_id)):
        
        temp_context.append('<PAD>')
        
      else:
        
        temp_context.append(self.document[doc_id]['i2s'][sent_id + i])
    
    return temp_context
  
  def get_context_words(self, doc_id, word, context_num):
    
    word_context = []
    
    doc_words = self.corpus.words(doc_id)
    
    word_occurrences = [i for i, w in enumerate(doc_words) if w == word]
    
    for occurrence in word_occurrences:
      
      temp_context = []
      
      for i in range(-context_num, context_num + 1):
        
        if occurrence + i < 0 or occurrence + i >= len(doc_words):
        
          temp_context.append('<PAD>')
        
        else:
        
          temp_context.append(doc_words[occurrence + i])
      
      word_context.append(temp_context)
    
    return word_context
        
  def get_wsd_context(self, dataset_type, word, word_sense, context_num = 0, is_word = False, is_sentence = True, is_document = False):
    
    if is_document == True:
      
      context = [self.get_document(doc_id) for doc_id in self.doc_ids[dataset_type] if word in self.document[doc_id]['wsd'] and \
                 word_sense in self.document[doc_id]['wsd'][word]]
    
    elif is_sentence == True:
      
      context = []
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            sentence_ids = self.document[doc_id]['wsd'][word][word_sense]
            
            for sent_id in sentence_ids:
              context.append(self.get_context_sentences(doc_id, sent_id, context_num))
    
    elif is_word == True:
      
      context = []
      
      for doc_id in self.doc_ids[dataset_type]:
        
        if word in self.document[doc_id]['wsd']:
          
          if word_sense in self.document[doc_id]['wsd'][word]:
            
            context.append(self.get_context_words(doc_id, word, context_num))
            
    return context
  
  def get_embeddings(self, word_to_index, embed_dim = 300, embed_type = 'glove'):

    if embed_type == 'glove':
      with open('glove_model.pckl', 'rb') as f:
        embed_model = pickle.load(f)
    else:
      pass

    embedding = torch.nn.Embedding(len(word_to_index) + 4, embed_dim)

    weight_matrix = np.zeros((len(word_to_index) + 4, embed_dim))
    
    weight_matrix[1:4, :] = np.random.randn(1, embed_dim)
    
    for word in word_to_index:
      weight_matrix[word_to_index[word] + 4, :] = embed_model[word].numpy()

    embedding.weight.data.copy_(torch.from_numpy(weight_matrix))
  
    return embedding
    
  def get_train_validation_split(self, split):
    
    self.doc_ids, self.vocab, self.i2w, self.w2i = {}, {}, {}, {}
    
    split_index = math.floor(self.num_docs * split)
    
    self.doc_ids['train'] = self.corpus.fileids()[0 : split_index]
    
    self.doc_ids['dev'] = self.corpus.fileids()[split_index : ]
    
    self.vocab['train'], self.w2i['train'], self.i2w['train'] = self.get_vocab(self.doc_ids['train'])
    
    self.vocab['dev'], self.w2i['dev'], self.i2w['dev'] = self.get_vocab(self.doc_ids['dev'])
    
  def preprocess_corpus(self):
    
    document = {}
        
    for k in self.corpus.fileids():
      
      document[k] = {}
      
      document[k]['i2s'] = self.get_index_2_sentence(k)
      
      document[k]['wsd'] = self.get_wsd_words(k)
      
    return document