# Semantics-Research-Project
Semantics Research Project

The script *dataset.py* contains the DatasetPreProcessor class.

# How to make use of this class

## Step 1 : To create an instance
dpp = DatasetPreProcessor(corpus = 'semcor')
corpus = 'semcor' / 'masc' (for now only 'semcor')

## Step 2 : To initialize the object with all the information you need
dpp.get_train_validation_split(split = <insert_value_here>)

## After Step 2, the object will contain the following members
### dpp.doc_ids - Dictionary containing the document IDs that belong to the training / validation sets
Keys -> 'train' or 'dev'
Returns: List of document IDs

### dpp.vocab - Dictionary containing the training / validation set vocabulary
Keys -> 'train' or 'dev'
Returns: Set of words

### dpp.w2i - Dictionary containing the word to index mapping for the training / validation sets
Keys -> 'train' or 'dev'
Returns: Dictionary with word as key and index as the value

### dpp.i2w - Dictionary containing the index to word mapping for the training / validation sets
Keys -> 'train' or 'dev'
Returns: Dictionary with index as key and word as the value

### dpp.document - Dictionary containing the information per document
#### Hierarchical Key Structure
document_id -> 'i2s'
document_id -> 'wsd'

#### To get the index to sentence mapping for a document
dpp.document[document_id]['i2s'][sentence_idx]

#### To get the words in the document that need to be disambiguated
dpp.document[document_id]['wsd']

#### To get the senses of a word that are present in a document
dpp.document[document_id]['wsd'][word]

#### To get the sentence indices that contain a particular word sense in the document
dpp.document[document_id]['wsd'][word_synset] -> List of sentence indices

## Helper functions
### To obtain the sentences of a document given its ID
dpp.get_document(document_id)

### To obtain the words of a document given its ID
dpp.get_words(document_id)

### To get the word, sentence or document-level context of a particular word sense
dpp.get_wsd_context(dataset_type, word, word_sense, context_num = 0, is_word = False, is_sentence = True, is_document = False)
dataset_type -> 'train' or 'dev'
word -> word to be disambiguated
word_sense -> synset representing a particular sense of the word
context_num -> the amount of context (word or sentence-level) before and after the word in consideration
(Default = 0)
is_word -> If you want word-level context
(Default = False)
is_sentence -> If you want sentence-level context
(Default = True)
is_document -> If you want document-level context
(Default = False)

Returns a list of lists where each list is a word, sentence or document-level context

### To get the word embeddings of the train / validation sets
dpp.get_embeddings(word_to_index_mapping, embed_dim = 300, embed_type = 'glove')
word_to_index_mapping -> dpp.w2i['train'] or dpp.w2i['dev']
embed_dim -> Embedding dimension (Default 300)
embed_type -> Type of Embedding to use (Default = 'glove') (For now only 'glove')

Returns a torch.nn.Embedding
