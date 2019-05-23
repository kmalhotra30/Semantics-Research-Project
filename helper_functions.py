from imports import *
def print_model_params(model):
  total = 0
  for name, p in model.named_parameters():
    total += np.prod(p.shape)
    print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
  print("\nTotal parameters: {}\n".format(total))
  
  
#Code to create checkpoint and save as .tar file   
    
def create_checkpoint(model_type,model,optimizer,train_avg_accuracy,train_avg_loss,epoch_no):

  if not os.path.exists('Checkpoints'):
    os.makedirs('Checkpoints')
  
  checkpoint_path_Model = "./Checkpoints/Model_" + model_type + "_" + str(epoch_no) + ".tar"
  checkpoint_Model = {
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "train_avg_accuracy": train_avg_accuracy,
              "train_avg_loss":train_avg_loss,
              "epoch":epoch_no
          }
  
  
  torch.save(checkpoint_Model, checkpoint_path_Model)


#Method to return max number of sentences and max sentence length over all documents in doc_ids list
def get_max_sent_length(dpp,dataset,context_window,masked_flag='unmasked'):
  
  max_sent_length = -1
  
  for index,row in dataset.iterrows():
    
    text_id = row['txt_id']
    sent_id = row['sentence_id']
    context , _, _,_ = dpp.get_context(dpp.text_id_mapping_reverse[text_id], sent_id, context_window)
    sentences = context[masked_flag]
    
    for sent in sentences :
      
      max_sent_length = max(max_sent_length,len(sent))
      
  return max_sent_length


def get_max_sent_length_sent_level(dpp,dataset,context_window,masked_flag='unmasked'):
  
  max_sent_length = -1
  
  for sent_id in dataset:
    
    txt_id_sent_id_inv_map = dpp.text_sent_mapping_reverse[sent_id]
    text_id = txt_id_sent_id_inv_map[0]
      
    context,labels,focus_sent_index,_ = dpp.get_context(text_id,sent_id,0)
    max_sent_length = max(max_sent_length,len(context['unmasked'][0]))
    
    
  return max_sent_length

class Doc_from_word :
  
  def __init__(self,datasetPreprocessorObj,dataset,context_window,masked_flag='unmasked'):
    
    self.dpp = datasetPreprocessorObj
    self.dataset = dataset
    self.context_window = context_window
    self.masked_flag = masked_flag
    self.max_no_of_sentences = 2 * context_window 
    
    if masked_flag == 'unmasked':
        self.max_no_of_sentences += 1
      
    self.max_sent_length = get_max_sent_length(datasetPreprocessorObj,dataset,context_window,masked_flag='unmasked')
    
      
  def create_doc_representation(self,word):

    sent_count_list = []

    doc_representation = [] #Initial empty list

    word_deepcopy = deepcopy(word)
    
    text_id = word_deepcopy[0].item()
    sent_id = word_deepcopy[1].item()
    token_id = word_deepcopy[2].item()

    context, labels ,focus_sent_index,sent_ids = self.dpp.get_context(self.dpp.text_id_mapping_reverse[text_id], sent_id, self.context_window)

    #focus_sentence_id = sent_ids[focus_sent_index] 
    
    labels_list = labels[self.masked_flag]
    sent_ids = sent_ids[self.masked_flag]

    labels_focus_sentence_list = labels_list[focus_sent_index]

    label_to_be_returned = labels_focus_sentence_list[token_id]
      
    context_list = context[self.masked_flag]

    sentences = context_list 

    count_sentences_in_doc = len(sentences)

    for sent_idx,sent in enumerate(sentences):

      sent_id = dpp.text_sent_mapping[(dpp.text_id_mapping_reverse[text_id],sent_ids[sent_idx])]

      sent_count_list.append(len(sent))
      sent_dummy_list = [] # Consists of word_idx for all words occuring in sent

      for word_idx,word in enumerate(sent):

        sent_dummy_list.append([text_id,sent_id,word_idx])

      #Making sentence list to the length max_sent_length
      #-1 denotes padding
      sent_dummy_list.extend([[text_id,sent_id,-1]] * (self.max_sent_length - len(sent)))

      #Adding sentence list to doc representation
      doc_representation.append(sent_dummy_list)

    #Making number of rows in doc representations = max_no_of_sentences

    sent_count_list.extend([1] * (self.max_no_of_sentences - len(sent_count_list)))
    sent_count_list = torch.LongTensor(sent_count_list)

    #Focus sentence labels are padded with -1
    #labels_focus_sentence_list.extend([-1] * (self.max_sent_length - len(labels_focus_sentence_list)))
    #text_id is a string 
    #sent_id is an integer
    #padding is deonoted by-1

    dummy_max_sent_length_pad_list = [[dpp.text_id_mapping["-1"],-1,-1]] * self.max_sent_length

    doc_representation.extend([dummy_max_sent_length_pad_list] * (self.max_no_of_sentences - len(doc_representation)))


    doc_representation = np.stack(doc_representation, axis=0)

    doc_representation = torch.LongTensor(doc_representation)
    #Focus sentence 1D Tensor (Padded)


    
    doc_dataset_dictionary = {'doc_rep':doc_representation,'count_sentences_in_doc':count_sentences_in_doc,\
'sent_count_list':sent_count_list,'label_to_be_returned':label_to_be_returned}

    return doc_dataset_dictionary


class Sent_from_word :
  
  def __init__(self,datasetPreprocessorObj,dataset,context_window,masked_flag='unmasked'):
    
    self.dpp = datasetPreprocessorObj
    self.dataset = dataset
    self.context_window = context_window
    self.masked_flag = masked_flag
    self.max_no_of_sentences = 2 * context_window 
    
    if masked_flag == 'unmasked':
        self.max_no_of_sentences += 1
      
    self.max_sent_length = get_max_sent_length(datasetPreprocessorObj,dataset,context_window,masked_flag='unmasked')
    print(self.max_sent_length)
    
      
  def create_sent_representation(self,word):

    
    sent_representation = [] #Initial empty list

    word_deepcopy = deepcopy(word)
    
    text_id = word_deepcopy[0].item()
    sent_id = word_deepcopy[1].item()
    token_id = word_deepcopy[2].item()

    context, labels ,focus_sent_index,sent_ids = self.dpp.get_context(self.dpp.text_id_mapping_reverse[text_id], sent_id, self.context_window)

    #focus_sentence_id = sent_ids[focus_sent_index] 
    
    labels_list = labels[self.masked_flag]
    sent_ids = sent_ids[self.masked_flag]

    
    labels_focus_sentence_list = labels_list[focus_sent_index]

    label_to_be_returned = labels_focus_sentence_list[token_id]
      
    context_list = context[self.masked_flag]

    sentences = context_list 
    
    focus_sentence = sentences[focus_sent_index]

    sent_count = len(focus_sentence)
    
    for word_idx,word in enumerate(focus_sentence):

      sent_representation.append([text_id,sent_id,word_idx])

    #Making sentence list to the length max_sent_length
    #-1 denotes padding
    sent_representation.extend([[text_id,sent_id,-1]] * (self.max_sent_length - len(focus_sentence)))
    
    sent_representation = torch.LongTensor(sent_representation)    

    return sent_representation,label_to_be_returned,sent_count

def create_targets_for_word_batch(word_batch,context_window):
  
  batch_size = word_batch.size(0)
  batch_t = torch.empty(batch_size,dtype=torch.long)
  
  for idx,word in enumerate(word_batch):
    
    word_deepcopy = deepcopy(word)
    txt_id = word_deepcopy[0].item()
    sent_id = word_deepcopy[1].item()
    token_id = word_deepcopy[2].item()
    
    _,labels,fsi,_ = dpp.get_context(txt_id,sent_id,context_window)
    
    batch_t[idx] = labels['unmasked'][fsi][token_id]
    
  batch_t = batch_t.to(device)
    
  return batch_t  

    

