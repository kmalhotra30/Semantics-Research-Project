#PyTorch Dataset class for words! This class is used for loading batched words
from imports import *
class wordDataset(Dataset):
  
    def __init__(self,datasetPreprocessorObj,dataset,context_window):
      
      #Parameters : datasetPreprocessorObj(dpp) 
      #           : dataset_type : - train/dev/test

      self.dpp = datasetPreprocessorObj
      self.dataset = dataset
      self.context_window = context_window
      
    def __len__(self):
  
      return len(self.dataset)
  
  
    def __getitem__(self, idx):
      
      word = self.dataset.iloc[idx]
   
      _,_,focus_sent_index,_ = dpp.get_context(word['txt_id'],word['sentence_id'],self.context_window)
                                                
      word_rep = torch.LongTensor([word['txt_id'],word['sentence_id'],word['word_id'],focus_sent_index])
        
      return word_rep

#Dataloder for sampling sentences from the Gao et al split (not NAACL)    
class sentDataset(Dataset):
    
    def __init__(self,datasetPreprocessorObj,dataset,context_window=0):
      
      #Parameters : datasetPreprocessorObj(dpp) 
      #           : dataset_type : - train/dev/test

      self.dpp = datasetPreprocessorObj
      #self.dataset_custom = dataset.sentence_id.unique()
      self.dataset = dataset
      self.context_window = context_window
      self.max_sent_length = get_max_sent_length_sent_level(dataset,0)
      
    def __len__(self):
  
      return len(self.dataset)
 
    def __getitem__(self, idx):
      
      sent_id = self.dataset[idx]

      txt_id_sent_id_inv_map = self.dpp.text_sent_mapping_reverse[sent_id]
      text_id = txt_id_sent_id_inv_map[0]
      
      context,labels,focus_sent_index,_ = dpp.get_context(text_id,sent_id,0)
      
     
      context_unmasked = context['unmasked']
      labels_unmasked = labels['unmasked']
      
      focus_sentence = context_unmasked[0]
      labels_focus = labels_unmasked[0]
      
      sent_count = len(focus_sentence)
      
      sent_dummy_list = []
      
      for word_idx,word in enumerate(focus_sentence):

        sent_dummy_list.append([dpp.text_id_mapping[text_id],sent_id,word_idx])

      #Making sentence list to the length max_sent_length
      #-1 denotes padding
      sent_dummy_list.extend([[dpp.text_id_mapping[text_id],sent_id,-1]] * (self.max_sent_length - len(focus_sentence)))
      labels_focus.extend([0] * (self.max_sent_length - len(focus_sentence)))

          
      sent_dummy_list = torch.LongTensor(sent_dummy_list)
      labels_focus = torch.LongTensor(labels_focus)
      
      return sent_dummy_list,labels_focus,sent_count

