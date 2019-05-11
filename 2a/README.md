# How to use the DatasetPreProcessor class in dataset.py
```python
# Make sure that the vuamc_corpus_train.csv, vuamc_corpus_test.csv and vuamc_glove_vectors.pckl are in the same folder as dataset.py
```
## Step 1

### Create an instance of this class
```python
dpp = DatasetPreProcessor(first_time = True) # If you have not created the word embedding pickle file
# or
dpp = DatasetPreProcessor(first_time = False) # If you have already created the word embedding pickle file
```
#### Variables created after performing this step
```python
dpp.corpus_name = 'vuamc' # Corpus name (For naming the files while saving)
dpp.vocab # Vocabulary of the entire corpus
dpp.texts # Dictionary containing information for each text in the corpus
  dpp.texts[<text_id>][<sent_id>]['token_list'] returns a list of tokens in the sentence <sent_id> belonging to text <text_id>
  dpp.texts[<text_id>][<sent_id>]['labels'] returns a list of labels for each token in the sentence <sent_id> belonging to text <text_id>
    0 indicates non-metaphor
    1 indicates metaphor
  <text_id> is a string
  <sent_id> is an integer
dpp.w2i # Dictionary mapping words to their indices
  dpp.w2i['<PAD>'] = 0
  dpp.w2i['<UNK>'] = 1
  dpp.w2i['<SOS>'] = 2
  dpp.w2i['<EOS>'] = 3
dpp.embeddings # torch.nn.Embedding containing the word embeddings (GloVe) following the index mapping of dpp.w2i
```
## Step 2
### To get the training, validation and test datasets
```python
dpp.get_original_split_datasets(train_dev_split = 0.7)
train_dev_split # fraction of the training set to split into training and validation sets
```
#### Variables created after performing this step
```python
dpp.original_dataset # Dictionary containing the original un-split training and test sets
  dpp.original_dataset['train'] # Original Unsplit training dataset
  dpp.original_dataset['test'] # Original test dataset
dpp.split_dataset # Dictionary containing the new split datasets
  dpp.split_dataset['train'] # Training set split from the original training dataset
  dpp.split_dataset['dev'] # Validation set split from the original training dataset
  dpp.split_dataset['test'] # Test set which is the same as the original test dataset
```

## Step 3
### To get the context sentences around a focus sentence from the text that it belongs to
```python
context, context_labels = dpp.get_context(text_id, sent_id, context_window)
text_id is a string
sent_id is an integer
context_window # Parameter to fetch context_window sentences before and after the focus sentence <sent_id>
```
#### Variables Returned
```python
context # Dictionary containing the masked and unmasked versions of the context of the focus sentence
  context['masked'] # List of Lists where each sub-list contains the tokens of the context sentences EXCLUDING those of the focus sentence
  context['unmasked'] # List of Lists where each sub-list contains the tokens of the context sentences INCLUDING those of the focus sentence
context_labels # Dictionary containing the metaphor labels of the masked and unmasked versions of the contexts of the focus sentence
  context_labels['masked'] # List of Lists where each sub-list contains the labels of each word in the context sentences EXCLUDING those of the focus sentence
  context_labels['unmasked'] # List of Lists where each sub-list contains the labels of each word in the context sentences INCLUDING those of the focus sentence
```