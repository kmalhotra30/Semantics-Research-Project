#Model to retrieve embeddings
from imports import *
class List_to_embedding(nn.Module):
  
  def __init__(self,dpp_obj,dim_embedding,text_sent_embedding_dict,glove_flag=False,key_for_dict = 'glove_elmo'):
    
    super(List_to_embedding, self).__init__()
    
    self.dim_embedding = dim_embedding
    self.text_sent_embedding_dict = text_sent_embedding_dict
    self.key_for_dict = key_for_dict
    self.dpp_obj = dpp_obj
    self.glove_flag = glove_flag
    
  def forward(self,input_tensor,**kwargs):
    
    #Input is a 2D tensor : Shape B X N
    #Output should be 3D : Shape B X N X D

    dimensions = list(input_tensor.size())
    
    if len(dimensions) == 2:
     
      tensor_containing_embedding = torch.zeros(dimensions[0],self.dim_embedding)
      
    else:
      
      tensor_containing_embedding = torch.zeros(dimensions[0],dimensions[1],self.dim_embedding)
      
      
    input_tensor_detached = deepcopy(input_tensor)
    
    for i in range(dimensions[0]):
      
      if len(dimensions) == 3: 
        
        for j in range(dimensions[2]):

          
          txt_id = self.dpp_obj.text_id_mapping_reverse[input_tensor_detached[i][j][0].item()]
          sent_id = input_tensor_detached[i][j][1].item()
          token_id = input_tensor_detached[i][j][2].item()

          if self.glove_flag == True:

            embedding = self.text_sent_embedding_dict[txt_id][sent_id][token_id][self.key_for_dict]
            embedding = embedding.narrow(1,0,self.dim_embedding)  

          else:

            embedding = self.text_sent_embedding_dict[txt_id][sent_id][token_id][self.key_for_dict]

          tensor_containing_embedding[i][j] = embedding
      
      elif len(dimensions) == 2:
        
        txt_id = self.dpp_obj.text_id_mapping_reverse[input_tensor_detached[i][0].item()]
        sent_id = input_tensor_detached[i][1].item()
        token_id = input_tensor_detached[i][2].item()
        
        if self.glove_flag == True:

          embedding = self.text_sent_embedding_dict[txt_id][sent_id][token_id][self.key_for_dict]
          embedding = embedding.narrow(1,0,self.dim_embedding)  

        else:

          embedding = self.text_sent_embedding_dict[txt_id][sent_id][token_id][self.key_for_dict]

                
        tensor_containing_embedding[i] = embedding


    return tensor_containing_embedding    
  
