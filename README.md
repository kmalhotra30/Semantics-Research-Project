# Semantics-Research-Project
Semantics Research Project

The script *dataset.py* contains the DatasetPreProcessor class.

# How to make use of this class

## Step 1 : To create an instance (loads with all the information required)
dpp = DatasetPreProcessor(corpus = 'semcor', first_time = False)

corpus = 'semcor' / 'masc' (for now only 'semcor')

first_time = True / False

Set to True if you are running for the first time

Set to False otherwise

Will load the per document information from 'semcor_document.pckl'

Will load the word embeddings from 'semcor_glove_vectors.pckl'

## Step 2 : To get the train / validation split
dpp.get_train_validation_split(split = <insert_value_here>)

## After Steps 1 and 2, the object will contain the following members
### dpp.doc_ids - Dictionary containing the document IDs that belong to the training / validation sets
Keys -> 'train' or 'dev'

Returns: List of document IDs

### dpp.vocab - Dictionary containing the vocabulary of the entire corpus

Returns: Set of words

### dpp.w2i - Dictionary containing the word to index mapping for the entire corpus vocabulary

Returns: Dictionary with word as key and index as the value

### dpp.embedding - Contains glove vector embeddings for each word in the vocabulary
Returns: torch.nn.Embedding (Make sure to supply it with torch.LongTensors)

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
dpp.document[document_id]['wsd'][word_synset_string] -> List of sentence indices

word_synset_string is of the form 'amicable.a.01'

## Helper functions
### To obtain the sentences of a document given its ID
dpp.get_document(document_id)

### To obtain the words of a document given its ID
dpp.get_words(document_id)

### To obtain the synset defintion given a synset string
dpp.get_definition(synset_string)

### To get the word, sentence or document-level context of a particular word sense
dpp.get_wsd_context(dataset_type, word, word_sense, context_num = 0, is_word = False, is_sentence = True, is_document = False)

dataset_type -> 'train' or 'dev'

word -> word to be disambiguated

word_sense -> synset_string representing a particular sense of the word (of the form 'amicable.a.01')

context_num -> the amount of context (word or sentence-level) before and after the word in consideration

(Default = 0)

is_word -> If you want word-level context

(Default = False)

Output: List of List of Lists 

First List is the overall collection

Second List consists of the word contexts from a document

Third List consists of the word contexts corresponding to each occurrence of the word in the document

is_sentence -> If you want sentence-level context

(Default = True)

Output: List of List of Lists 

First List is the overall collection

Second List consists of the sentence contexts from a document

Third List consists of the sentence contexts corresponding to each occurrence of the word in the document

is_document -> If you want document-level context

(Default = False)

Returns a list of lists where each list corresponds to the entire document

### To get the (pruned) word embeddings of the train / validation sets
dpp.get_embeddings(word_to_index_mapping, embed_dim = 300, embed_type = 'glove', first_time = False)

word_to_index_mapping -> dpp.w2i

embed_dim -> Embedding dimension (Default 300)

embed_type -> Type of Embedding to use (Default = 'glove') (For now only 'glove')

first_time -> True -> If running this function for the first time

first_time -> False -> Load the word embeddings from the corresponding pickle file

Returns a torch.nn.Embedding
