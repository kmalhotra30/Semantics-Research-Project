from imports import *
class GRU(nn.Module):
  
  def __init__(self,hidden_dim,embedding_dim):
    
    super(GRU,self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    
    
  def forward(self,input_tensor,input_lengths):
   
    #Updating batch_size 
    
    self.batch_size = input_tensor.size(0)
    
    idx_input_sort = torch.argsort(-1 * input_lengths)
    input_lengths = input_lengths[idx_input_sort]
    
    #Sorting input_tensor on the basis of lengths (decreasing lenghts)
    input_tensor = input_tensor[idx_input_sort]
    
    #Unsort indices
    idx_input_unsort = torch.argsort(idx_input_sort)
    
    
    #Packing 
    input_tensor_packed= torch.nn.utils.rnn.pack_padded_sequence(input_tensor, input_lengths,batch_first=True)
    
    #GRU forward pass
    output_packed, _ = self.gru(input_tensor_packed)
  
    #Unpacking sequence 
    output,_ = torch.nn.utils.rnn.pad_packed_sequence(output_packed,batch_first=True)
    
    #Unsorting 
    output = output[idx_input_unsort]
    
    
    
    
    #Todo 
    #Creating Mask 
   
    #Mask = torch.zeros(self.batch_size,dim_1_len,dim_2_len)
    #dim_1_len = output.size(1)
    #dim_2_len = output.size(2)
   
    return output
    
class LSTM(nn.Module):
  
  def __init__(self,hidden_dim,embedding_dim):
    
    super(LSTM,self).__init__()
    self.hidden_dim = hidden_dim
    self.embedding_dim = embedding_dim
    self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
    self.ip_dropout = nn.Dropout(0.5)
    
    
  def forward(self,input_tensor,input_lengths):
   
    #Updating batch_size 
    
   #input_tensor = self.ip_dropout(input_tensor)
    
    self.batch_size = input_tensor.size(0)
    
    idx_input_sort = torch.argsort(-1 * input_lengths)
    input_lengths = input_lengths[idx_input_sort]
    
    #Sorting input_tensor on the basis of lengths (decreasing lenghts)
    input_tensor = input_tensor[idx_input_sort]
    
    #Unsort indices
    idx_input_unsort = torch.argsort(idx_input_sort)
    
    
    #Packing 
    input_tensor_packed= torch.nn.utils.rnn.pack_padded_sequence(input_tensor, input_lengths,batch_first=True)
    
    #LSTM forward pass
    output_packed, _ = self.lstm(input_tensor_packed)
  
    #Unpacking sequence 
    output,_ = torch.nn.utils.rnn.pad_packed_sequence(output_packed,batch_first=True)
    
    #Unsorting 
    output = output[idx_input_unsort]
    
    
    #Todo 
    #Creating Mask 
   
    #Mask = torch.zeros(self.batch_size,dim_1_len,dim_2_len)
    #dim_1_len = output.size(1)
    #dim_2_len = output.size(2)
   
    return output


class Attention_Module(nn.Module):
  
  def __init__(self,encoder_obj,hidden_dim):
    
    
    super(Attention_Module,self).__init__()
    self.hidden_dim = hidden_dim
    self.encoder_obj = encoder_obj
    
    #Defining parameters/layers and non linearities
    
    self.pre_attention = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
    
    self.context_weights = nn.Parameter(torch.randn(hidden_dim,1))
    self.ip_dropout = nn.Dropout(0.2)
    
  
  def forward(self,input_tensor,input_lengths):
    
    #Shape of Input : B X N X embed_dim
    
    #Pass through encoder
    
    
    #input_tensor = self.ip_dropout(input_tensor)
    
    h = self.encoder_obj(input_tensor,input_lengths)
    
    #Shape : B X N X hidden_dim
    
    u = self.pre_attention(h)
    
    #Shape : B X N X hidden_dim
    
    alpha_un_normalized = torch.matmul(u,self.context_weights)
    
    #Shape : B X N X 1
    
    alpha_normalized = nn.functional.softmax(alpha_un_normalized,dim=1)
    
    #Shape : B X N X 1
    
    #Element Wise multiplication of alpha with encoding
    
    alpha_element_wise_product = alpha_normalized * h
    
    #Shape : B X N X hidden_dim
    
    #Sentence/doc encoding creation by attention
    
    encoding = alpha_element_wise_product.sum(1)
    
    return encoding,h

class HAN_Model(nn.Module):
  
  def __init__(self,word_hidden_dim,embedding_model_obj,word_encoder_obj,sent_encoder_obj,word_attention_module_obj,sent_attention_module_obj,device):
    
    super(HAN_Model,self).__init__()
    
    self.word_encoder_obj = word_encoder_obj
    self.sent_encoder_obj = sent_encoder_obj
    
    self.word_attention_module_obj = word_attention_module_obj
    self.sent_attention_module_obj = sent_attention_module_obj
    self.embedding_model_obj = embedding_model_obj
    
    self.word_hidden_dim = word_hidden_dim
    
    self.device = device
    
  def forward(self,input_doc_batch,doc_len,input_lengths,focus_sentence_word_list):

    #Input_lengths is of format : tuple(no_of_sentences,List(sent lengths))

    self.batch_size = input_doc_batch.size(0)
    seq_len = input_doc_batch.size(1)

    sent_encodings = torch.zeros(self.batch_size,seq_len,2 *self.word_hidden_dim) # Will serve as batch for sentences
    word_activation_batch = torch.zeros(self.batch_size,2 *self.word_hidden_dim).to(device)
    
    for idx,doc in enumerate(input_doc_batch):

      doc = self.embedding_model_obj(doc).to(self.device)
      input_lengths[idx] = torch.LongTensor(input_lengths[idx])
      
      sent_rep,word_activations = self.word_attention_module_obj(doc,input_lengths[idx])
      
      focus_sent_id = focus_sentence_word_list[idx][0]
      focus_token_id = focus_sentence_word_list[idx][1]
      
      word_activation_batch[idx] = word_activations[focus_sent_id][focus_token_id]
      
      sent_encodings[idx] = sent_rep


    docs_to_be_encoded = torch.Tensor(sent_encodings)
    docs_to_be_encoded = docs_to_be_encoded.to(self.device)
    doc_rep,_  = self.sent_attention_module_obj(docs_to_be_encoded,doc_len)

    return doc_rep,word_activation_batch 

class DOC_Bow_Model(nn.Module):
  
  def __init__(self,embedding_model_obj,dim_embedding,device):
    
    super(DOC_Bow_Model,self).__init__()
    
    self.device = device
    self.embedding_model_obj = embedding_model_obj
    self.dim_embedding = dim_embedding
    
  def forward(self,input_doc_batch,doc_len,input_lengths):
    
    batch_size = input_doc_batch.size(0)
    
    output_tensor = torch.zeros(batch_size,self.dim_embedding)
    
    for idx,doc in enumerate(input_doc_batch):

      doc = self.embedding_model_obj(doc).to(self.device)
      ip_len = torch.LongTensor(input_lengths[idx])
      
      limit_len = doc_len[idx]
      
      true_sent_lens = ip_len.narrow(0,0,limit_len)
      
      length = torch.sum(true_sent_lens)
      
      bow_sum = (torch.sum(torch.sum(doc,1),0))
      
      
      output_tensor[idx] = bow_sum/limit_len
      
    return output_tensor.to(self.device)
  
class Classification_Model(nn.Module):
  
  def __init__(self,input_dim,num_classes):
  
    super(Classification_Model,self).__init__()
    
    self.input_dim = input_dim
    self.num_classes = num_classes
    
    self.classification_layer = nn.Linear(self.input_dim,self.num_classes)
    self.ip_dropout = nn.Dropout(0.1) 
    
  
  def forward(self,input_tensor):
    
    #input_tensor = self.ip_dropout(input_tensor)
    return self.classification_layer(input_tensor)



class Discourse_and_Classification(nn.Module):
  
  def __init__(self,embedding_model,document_encoding_model,classification_model,classification_dim):
  
    super(Discourse_and_Classification,self).__init__()
    
    self.document_encoding_model = document_encoding_model
    
    self.classification_model = classification_model
    self.doc_max_sent_length = 0
    self.doc_max_no_of_sentences = 0
    self.classification_dim = classification_dim
    self.embedding_model = embedding_model
  

  def forward(self,doc_creation_class_obj,batch_of_words):
    
    self.doc_creation_class_obj = doc_creation_class_obj
     

    self.doc_max_sent_length = doc_creation_class_obj.max_sent_length 
    self.doc_max_no_of_sentences = doc_creation_class_obj.max_no_of_sentences
    
    batch_size = batch_of_words.size(0)
    
    
    batch_of_words_embedded = self.embedding_model(batch_of_words).to(device)
    
    
    doc_rep_batch = torch.ones(batch_size,self.doc_max_no_of_sentences,self.doc_max_sent_length,3)
    count_sentences_in_doc_batch = torch.ones(batch_size,dtype=torch.long)
    sent_count_list_batch = torch.ones(batch_size,self.doc_max_no_of_sentences,dtype=torch.long)
    label_batch = torch.ones(batch_size,dtype=torch.long)
    focus_sentence_word_list = torch.ones(batch_size,2,dtype=torch.long)
    
    for idx,word in enumerate(batch_of_words):
      
      focus_sentence_id = word[3]
      token_id = word[2]
      
      doc_dataset_dictionary = self.doc_creation_class_obj.create_doc_representation(word)
      
      doc_rep_batch[idx] = doc_dataset_dictionary['doc_rep']
      count_sentences_in_doc_batch[idx] = doc_dataset_dictionary['count_sentences_in_doc']
      sent_count_list_batch[idx] = doc_dataset_dictionary['sent_count_list']
      
      focus_sentence_word_list[idx] = torch.LongTensor([focus_sentence_id,token_id])
      
      
      label_batch[idx] = doc_dataset_dictionary['label_to_be_returned']
      
    doc_rep_batch = doc_rep_batch.to(device)
    focus_sentence_word_list = focus_sentence_word_list.to(device)
    
    label_batch = label_batch.to(device)
    
    #Forward pass with Document encoding model
    discourse_representation,word_activation = self.document_encoding_model(doc_rep_batch,count_sentences_in_doc_batch,sent_count_list_batch,focus_sentence_word_list)
    
    #creating batch with discourse + embedding or Bi-lstm concatenated
    
  	#For discourse + embedding
    #batch_for_classifier = torch.cat((discourse_representation,batch_of_words_embedded),1)
    
    #For discourse + bi-lstm

    batch_for_classifier = torch.cat((discourse_representation,word_activation),1)
    
      
    batch_for_classifier = batch_for_classifier.to(device)
    classification_activation = self.classification_model(batch_for_classifier)
    
    return nn.functional.log_softmax(classification_activation,dim=1), label_batch


class justClassifier_Model(nn.Module):
  
  def __init__(self,embedding_model,classifier_model):
    
    super(justClassifier_Model,self).__init__()
    
    self.embedding_model = embedding_model
    self.classifier_model = classifier_model
    
  def forward(self,word_batch):
    
    word_batch_embedded = self.embedding_model(word_batch).to(device)
    return self.classifier_model(word_batch_embedded)

class Gao_original(nn.Module):
  
  def __init__(self,embedding_model,sent_encoding_model,classification_model,classification_dim):
    
    super(Gao_original,self).__init__()
    
    self.sent_encoding_model = sent_encoding_model
    
    self.classification_model = nn.Linear(1924,2)
    self.doc_max_sent_length = 0
    self.doc_max_no_of_sentences = 0
    self.classification_dim = classification_dim
    self.embedding_model = embedding_model
    self.lstm = nn.LSTM(1324, 300, bidirectional=True, batch_first=True)
    self.input_to_lstm_dropout = nn.Dropout(0.5)
    self.input_to_classification_dropout = nn.Dropout(0.1)
    
    
  def forward(self,input_tensor,input_lengths):


    batch_size = input_tensor.size(0)


    input_tensor_deepcopy = deepcopy(input_tensor)
    input_tensor_embedded = self.embedding_model(input_tensor_deepcopy).to(device)
    input_tensor_to_lstm = deepcopy(input_tensor_embedded)
    #Forward pass with Sentence encoding model

    #definng LSTm operation here

    idx_input_sort = torch.argsort(-1 * input_lengths)
    input_lengths = input_lengths[idx_input_sort]

    #Sorting input_tensor on the basis of lengths (decreasing lenghts)
    input_tensor_to_lstm = input_tensor_to_lstm[idx_input_sort]

    #Unsort indices
    idx_input_unsort = torch.argsort(idx_input_sort)


    #Packing 
    #input_tensor_embedded = self.input_to_lstm_dropout(input_tensor_embedded)

    input_tensor_packed= torch.nn.utils.rnn.pack_padded_sequence(input_tensor_to_lstm, input_lengths,batch_first=True)

    #LSTM forward pass
    output_packed, _ = self.lstm(input_tensor_packed)

    #Unpacking sequence 
    output,_ = torch.nn.utils.rnn.pad_packed_sequence(output_packed,batch_first=True)

    #Unsorting 
    word_activations = output[idx_input_unsort]
    
    batch_size = word_activations.size(0)
    seq_len = word_activations.size(1)
    
    
    
    concat_emeddings = torch.zeros(batch_size,seq_len,1924).to(device)
    
    for i in range(batch_size):
      
      for j in range(seq_len):
        
        concat_emeddings[i][j] = torch.cat((word_activations[i][j],input_tensor_embedded[i][j]),0)
        
        
        
    
    #word_activations = self.input_to_classification_dropout(word_activations)

#     packed = nn.utils.rnn.pack_padded_sequence(input_tensor_embedded, input_lengths, batch_first=True, enforce_sorted=False)

#     output,_ = self.lstm(packed)

#     word_activations = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)[0]

    #classification_activations = self.classification_model(word_activations)
    classification_activations = self.classification_model(concat_emeddings)
  
    

    
    return nn.functional.log_softmax(classification_activations, dim=2)


class Gao(nn.Module):
  
  def __init__(self,embedding_model,sent_encoding_model,classification_model,classification_dim):
  
    super(Gao,self).__init__()
    
    self.sent_encoding_model = sent_encoding_model
    
    self.classification_model = nn.Linear(classification_dim,2)
    self.doc_max_sent_length = 0
    self.doc_max_no_of_sentences = 0
    self.classification_dim = classification_dim
    self.embedding_model = embedding_model
    self.lstm = nn.LSTM(1324, 300, bidirectional=True, batch_first=True)
    
  

  def forward(self,doc_creation_class_obj,batch_of_words):
    
    self.doc_creation_class_obj = doc_creation_class_obj
     

    self.doc_max_sent_length = doc_creation_class_obj.max_sent_length 
    self.doc_max_no_of_sentences = doc_creation_class_obj.max_no_of_sentences
    
    batch_size = batch_of_words.size(0)
    
    
    batch_of_words_embedded = self.embedding_model(batch_of_words).to(device)
    
    
    sent_rep_batch = torch.ones(batch_size,self.doc_max_sent_length,3)
    sent_count_list_batch = torch.ones(batch_size,dtype=torch.long)
    label_batch = torch.ones(batch_size,dtype=torch.long)
    focus_sentence_word_list = torch.ones(batch_size,2,dtype=torch.long)
    
    for idx,word in enumerate(batch_of_words):
      
      focus_sentence_id = word[3]
      token_id = word[2]
      
      
      sent_representation,label_to_be_returned,sent_count = self.doc_creation_class_obj.create_sent_representation(word)
      
      
      sent_rep_batch[idx] = sent_representation
      sent_count_list_batch[idx] = sent_count
      
      focus_sentence_word_list[idx] = torch.LongTensor([focus_sentence_id,token_id])
      
      
      label_batch[idx] = label_to_be_returned
      
    sent_rep_batch_embedded = self.embedding_model(sent_rep_batch).to(device)
    focus_sentence_word_list = focus_sentence_word_list.to(device)
    
    label_batch = label_batch.to(device)
    
    #Forward pass with Sentence encoding model
    
    #definng LSTm operation here
    
    
    
    
    idx_input_sort = torch.argsort(-1 * sent_count_list_batch)
    sent_count_list_batch = sent_count_list_batch[idx_input_sort]
    
    #Sorting input_tensor on the basis of lengths (decreasing lenghts)
    sent_rep_batch_embedded = sent_rep_batch_embedded[idx_input_sort]
    
    #Unsort indices
    idx_input_unsort = torch.argsort(idx_input_sort)
    
    
    #Packing 
    input_tensor_packed= torch.nn.utils.rnn.pack_padded_sequence(sent_rep_batch_embedded, sent_count_list_batch,batch_first=True)
    
    #LSTM forward pass
    output_packed, _ = self.lstm(input_tensor_packed)
  
    #Unpacking sequence 
    output,_ = torch.nn.utils.rnn.pad_packed_sequence(output_packed,batch_first=True)
    
    #Unsorting 
    word_activations = output[idx_input_unsort]
    
    
    
    
    
    #word_activations = self.sent_encoding_model(sent_rep_batch_embedded,sent_count_list_batch)
    
    batch_for_classifier = torch.ones(batch_size,self.classification_dim)
    
    for idx in range(batch_size):
      
      #batch_for_classifier[idx] = torch.cat((discourse_representation[idx],word_activation[idx]),0)
      
      token_id = focus_sentence_word_list[idx][1]
      
      batch_for_classifier[idx] = word_activations[idx][token_id]
      
    batch_for_classifier = batch_for_classifier.to(device)
    
    return self.classification_model(batch_for_classifier), label_batch
    
    
    
    
   
