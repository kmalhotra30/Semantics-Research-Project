# Research Project for the course Statistical Methods for Natural Language Semantics
This repository contains the code for the Research Project of the course Statistical Methods for Natural Language Semantics. The goal of this project was to model discourse in metaphor resolution.

# Executing the code :

### 
Note : Information about Elmo weights and options can be viewed at (https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md)
- main.py : This file accepts the following flags :- 
    - --w2i_file : Location of word to index json file.
    - --i2w_file : Location of index to word json file.
    - --glove_filename : Name of the pickle file con'HAN_LSTMt,'VA_LSTMa for baseline 3, vanilla attention and Hierarchical Attentionining glove embeddings
    - --elmo_weights : Name of Weights file of ELMO.  for baseline 3, vanilla attention and Hierarchical Attention
    - --elmo_options : Name of options file of ELMO. 
    - --train_steps : Number of batch iterations.
    - --batch_size : Batch size.
    - --model_type : Can be either 'BL3_LSTM','HAN_LSTM','VA_LSTM' for baseline 3, Vanilla Attention and Hierarchical Attention.
    - --repetitions : Number of times model is to be trained and evaluated.
    - --context_window : Can be 0,1,2,3 etc.
    - --sort_data : 1 to sort data for faster training and 0 otherwise
    - --meta_folder : Path of folder containing the train,test,dev data
    - --meta_train : Name of the Training data file
    - --meta_valid : Name of the Validation data file
    - --meta_test : Name of the Test data file

    Flags pertaining to hyper-paramaters have been omitted in the description as it is desirable to not change the hyperparameters.
    
    Demo command to run the code :
    ```sh
    $ python3 "main.py" --meta_folder "VUAsequence/" --w2i_file "embeddings/w2i.json" --i2w_file "embeddings/i2w.json" --glove_filename "embeddings/embeddings.pickle" --elmo_weights "cached_elmo/weights_1024.hdf5" --elmo_options "cached_elmo/options_1024.json" --meta_train "train.csv" --meta_val "valid.csv" --meta_test "test.csv" --lr 0.005 --train_steps 1500 --dropout1 0.5 --dropout2 0.0 --dropout3 0.1 --num_layers 1 --hidden_size 100 --batch_size 64 --context_window 1 --model_type "VA_LSTM" 
 - analysis.py : It accepts the same set of arguements as main.py. This file loads a trained model and executes the same on test data. Further it produces a csv file containg various results which can be used to perform analysis.       
