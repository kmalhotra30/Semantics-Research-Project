from imports import *
from dataloaders import *
from datasetPreProcessor_classes import *
from embedding_model import *
from helper_functions import *
from helper_functions import Doc_from_word
from models import *

nltk.download('punkt')

dpp = DatasetPreProcessor()
dpp.pre_process_corpus()
dpp.get_mappings()
dpp.get_word_to_index()
dpp.get_embeddings('./', first_time = False)
dpp.get_original_split_datasets('./', first_time = False)
dpp.get_elmo_embedding_dict('./gdrive/My Drive/', first_time = False)


#For replicating Gao code

# dpp = DatasetPreProcessorGao()
# dpp.get_datasets()
# dpp.pre_process_corpus()
# dpp.get_mappings()
# dpp.get_embeddings('./', first_time = False)
# dpp.get_elmo_embedding_dict('./gdrive/My Drive/', first_time = False)
# train, dev, test = dpp.get_dataset_sent_ids()

embed_dataset_dict = dpp.elmo_embed_dict

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'

#Setting seed
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


config = {}

#Training Config
config['batch_size_han'] = 64
config['lr'] = 0.1
config['optimizer'] = 'Adam'
config['weight_decay'] = 0
config['momentum'] = 0.9
config['max_norm'] = 5.0

#Han Model Config
config['word_embedding_han_dim'] = 1324
#config['word_embedding_han_dim'] = 300

config['word_encoder_han'] = 'gru'
config['sent_encoder_han'] = 'gru'
config['word_hidden_han_dim'] = 300
config['sent_hidden_han_dim'] = 300
 

#Classification Model config
config['classification_model_num_classes'] = 2
config['classification_model_batch_size'] = 64


train = dpp.split_dataset['train']
dev = dpp.split_dataset['dev']
test = dpp.split_dataset['test']

# For replicating Gao

# sentTrainDataset = sentDataset(dpp,train,2)
# sentDevDataset = sentDataset(dpp,dev,2)
# sentTestDataset = sentDataset(dpp,test,2)


trainDataset = wordDataset(dpp,train,2)
devDataset = wordDataset(dpp,dev,2)
testDataset = wordDataset(dpp,test,2)


#For replicating Gao 

# sentTrainDataloader = DataLoader(sentTrainDataset,batch_size = config['batch_size_han'],
#                                 shuffle=True,num_workers=1)
# sentDevDataloader = DataLoader(sentDevDataset,batch_size = config['batch_size_han'],
#                                 shuffle=True,num_workers=1)

# sentTestDataloader = DataLoader(sentTestDataset,batch_size = config['batch_size_han'],
#                                 shuffle=True,num_workers=1)

wordTrainDataloader = DataLoader(trainDataset, batch_size=config['batch_size_han'],
                           shuffle=False, num_workers=4)

wordDevDataloader = DataLoader(devDataset, batch_size=config['batch_size_han'],
                           shuffle=False, num_workers=4)

wordTestDataloader = DataLoader(testDataset, batch_size=config['batch_size_han'],
                           shuffle=False, num_workers=4)


#Declaring models

embed_model_obj = List_to_embedding(dpp,config['word_embedding_han_dim'],embed_dataset_dict,glove_flag=False)
embed_model_obj = embed_model_obj.to(device)

w_encoder = LSTM(config['word_hidden_han_dim'],config['word_embedding_han_dim'])
w_encoder = w_encoder.to(device)

word_attention_module_obj = Attention_Module(w_encoder,2* config['word_hidden_han_dim'])
word_attention_module_obj = word_attention_module_obj.to(device)

s_Encoder = LSTM(config['sent_hidden_han_dim'],2 * config['word_hidden_han_dim'])
s_Encoder = s_Encoder.to(device)

sent_attention_module_obj = Attention_Module(s_Encoder,2*config['sent_hidden_han_dim'])
sent_attention_module_obj = sent_attention_module_obj.to(device)
    
han_model = HAN_Model(config['word_hidden_han_dim'],embed_model_obj,w_encoder,s_Encoder,word_attention_module_obj,sent_attention_module_obj,device)
han_model = han_model.to(device)


#Different configurations can be tried i.e Discourse + Bi-Lstm or Discourse + elmo + glove

#classification_model_obj = Classification_Model(2 * config['word_hidden_han_dim'] + config['word_embedding_han_dim'],config['classification_model_num_classes'])
classification_model_obj = Classification_Model(2 * config['word_hidden_han_dim'],config['classification_model_num_classes'])

classification_model_obj = classification_model_obj.to(device)

doc_from_word_obj_train = Doc_from_word(dpp,train,2,masked_flag='unmasked')
doc_from_word_obj_dev = Doc_from_word(dpp,dev,2,masked_flag='unmasked')
doc_from_word_obj_test = Doc_from_word(dpp,test,2,masked_flag='unmasked')

#For replicating Gao

# Sent_from_word_obj_train = Sent_from_word(dpp,train,2,masked_flag='unmasked')
# Sent_from_word_obj_dev = Sent_from_word(dpp,dev,2,masked_flag='unmasked')
# Sent_from_word_obj_test = Sent_from_word(dpp,test,2,masked_flag='unmasked')

discourse_and_classification_obj = Discourse_and_Classification(embed_model_obj,doc_bow,classification_model_obj,2 * config['word_embedding_han_dim'])

discourse_and_classification_obj = discourse_and_classification_obj.to(device)

print_model_params(discourse_and_classification_obj)


def train_model(model_type,model,doc_from_word_obj_train,doc_from_word_obj_dev,doc_from_word_obj_test,trainDataloader,devDataloader,testDataloader,writer = SummaryWriter('runs/Metahphor')):
  
  
  epoch_no = 1
  learn_rate = 0.005
  weight_decay = config['weight_decay']
  dev_set_loss_min = 10000000.0
  best_model_epoch = 1
  momentum = config['momentum']
  
  batch_train_scores = list()
  batch_train_losses = list()
  batch_train_accuracies = list()

  validation_scores = list()
  validation_losses = list()
  validation_accuracies = list()
  
  #criterion = nn.CrossEntropyLoss()

  criterion = nn.NLLLoss()
  #For experimenting with weighted loss
  #criterion = nn.NLLLoss(weight=torch.Tensor([0.3,0.7]).to(device))
   

  #optimizer = optim.Adam(model.parameters(),lr=learn_rate,weight_decay=weight_decay)
  optimizer = optim.Adam(model.parameters(),lr=learn_rate,weight_decay=weight_decay)
     
  while(True):
    

#     if epoch_no >= 11:
#       for param_group in optimizer.param_groups:
#         param_group['lr'] = 0.0001
      
    model.train()

    #For replicating Gao
    #Gao_original_obj.train()
    
    batch_train_losses.append([])
    batch_train_accuracies.append([])
    
    scores_dict = {'precision_0':[],'recall_0':[],'f_score_0' : [],'precision_1':[],'recall_1':[],'f_score_1' : []}
          
    all_predictions = torch.empty(0,dtype=torch.long).to(device)
    all_ground_truth = torch.empty(0,dtype=torch.long).to(device)
    
    #For replicating Gao

#     for idx,(sent_d,labels,count) in enumerate(sentTrainDataloader): 

        
#       sent_dummy_list,labels_focus,sent_count = sent_d,labels,count
#       sent_dummy_list = sent_dummy_list.to(device)
#       batch_t = labels_focus.to(device)
      
# #       sent_dummy_list,labels_focus,sent_count = sent_d.narrow(0,0,2),labels.narrow(0,0,2),count.narrow(0,0,2)
# #       sent_dummy_list = sent_dummy_list.to(device)
# #       batch_t = labels_focus.to(device)
      
#       forward_pass_classification_model = Gao_original_obj(sent_dummy_list,sent_count) 
      
#       no_of_samples_in_batch = forward_pass_classification_model.size(0)
#       seq_len = forward_pass_classification_model.size(1)
      
#       batch_t = batch_t.narrow(1,0,seq_len)

    
    for idx,word_batch in enumerate(trainDataloader):
    
        
      
      word_batch = word_batch.to(device)
      
      forward_pass_classification_model,batch_t = model(doc_from_word_obj_train,word_batch)

      #Compute loss

      loss = criterion(forward_pass_classification_model.contiguous().view(-1,2),batch_t.contiguous().view(-1))

      predictions = forward_pass_classification_model.argmax(1)
      #predictions = forward_pass_classification_model.argmax(2)
      no_of_predictions = predictions.size(0)
      
      all_predictions = torch.cat((all_predictions,predictions.contiguous().view(-1)),0)
      all_ground_truth = torch.cat((all_ground_truth,batch_t.contiguous().view(-1)),0)
      
      
      batch_t_deepcopy = deepcopy(batch_t.contiguous().view(-1))
      predictions_deepcopy = deepcopy(predictions.contiguous().view(-1))
      
      
      batch_accuracy = accuracy_score(batch_t_deepcopy.detach().cpu().numpy(),predictions_deepcopy.detach().cpu().numpy())
      batch_train_accuracies[epoch_no - 1].append(batch_accuracy)

      batch_train_losses[epoch_no - 1].append(loss.item())
      
      
      optimizer.zero_grad()
      loss.backward()

      optimizer.step()

      
       
    precision,recall,f_score,_ = precision_recall_fscore_support(all_ground_truth.detach().cpu().numpy(), all_predictions.detach().cpu().numpy(), labels = [0,1],average=None)
    scores_dict['precision_0'] = precision[0]
    scores_dict['recall_0'] = recall[0]
    scores_dict['f_score_0'] = f_score[0]
    
    scores_dict['precision_1'] = precision[1]
    scores_dict['recall_1'] = recall[1]
    scores_dict['f_score_1'] = f_score[1]

    batch_train_scores.append(scores_dict)
    
    
    train_avg_accuracy = np.mean(batch_train_accuracies[epoch_no - 1])
    train_avg_loss = np.mean(batch_train_losses[epoch_no - 1])
    
    
    print("Average Train Accuracy at Epoch " + str(epoch_no)+ ": ",train_avg_accuracy)
    print("Average Train Loss at Epoch " + str(epoch_no)+ ": " ,train_avg_loss)
    
    print("Precision for class 0 at Epoch " + str(epoch_no) + ":",scores_dict['precision_0'])
    print("Recall for class 0 at Epoch " + str(epoch_no) + ":",scores_dict['recall_0'])
    print("F-Score for class 0 at Epoch " + str(epoch_no) + ":",scores_dict['f_score_0'])
    
    print("Precision for class 1 at Epoch " + str(epoch_no) + ":",scores_dict['precision_1'])
    print("Recall for class 1 at Epoch " + str(epoch_no) + ":",scores_dict['recall_1'])
    print("F-Score for class 1  at Epoch " + str(epoch_no) + ":",scores_dict['f_score_1'])
       
    create_checkpoint(model_type,model,optimizer,train_avg_accuracy,train_avg_loss,epoch_no)


    writer.add_scalar('Average Train Acccuracy', train_avg_accuracy, epoch_no)
    writer.add_scalar('Average Train Loss', train_avg_loss, epoch_no)
    
    #Evaluating on dev set

    print("")
    model.eval()
    #Gao_original_obj.eval()
    
    scores_dict = {'precision_0':[],'recall_0':[],'f_score_0' : [],'precision_1':[],'recall_1':[],'f_score_1' : []}
    
    #print(all_predictions.size())
    all_predictions = torch.empty(0,dtype=torch.long).to(device)
    all_ground_truth = torch.empty(0,dtype=torch.long).to(device)
    
    dev_acc = 0
    dev_loss = 0
  
    counter = 0
   
   	#For replicating Gao

#     for idx,(sent_d,labels,count) in enumerate(sentDevDataloader): 

#       sent_dummy_list,labels_focus,sent_count = sent_d,labels,count
#       sent_dummy_list = sent_dummy_list.to(device)
#       batch_t = labels_focus.to(device)

#       forward_pass_classification_model = Gao_original_obj(sent_dummy_list,sent_count) 
      
#       no_of_samples_in_batch = forward_pass_classification_model.size(0)
#       seq_len = forward_pass_classification_model.size(1)
      
#       batch_t = batch_t.narrow(1,0,seq_len)
      
#       no_of_samples_in_batch = forward_pass_classification_model.size(0)
      

    for idx,word_batch in enumerate(devDataloader):
      
    
    #for i in range(1):
      
      #word_batch = devDataPruned
      #batch_t = create_targets_for_word_batch(word_batch,2).to(device)
      
      word_batch = word_batch.to(device)
      
      forward_pass_classification_model,batch_t = model(doc_from_word_obj_dev,word_batch)

      
      #Compute loss
      
      
      predictions = forward_pass_classification_model.argmax(1)
      #predictions = forward_pass_classification_model.argmax(2)
      no_of_predictions = predictions.size(0)
      
      all_predictions = torch.cat((all_predictions,predictions.contiguous().view(-1)),0)
      all_ground_truth = torch.cat((all_ground_truth,batch_t.contiguous().view(-1)),0)

      loss = criterion(forward_pass_classification_model.contiguous().view(-1,2),batch_t.contiguous().view(-1))

      
      all_predictions = torch.cat((all_predictions,predictions.contiguous().view(-1)),0)
      all_ground_truth = torch.cat((all_ground_truth,batch_t.contiguous().view(-1)),0)
      
      batch_t_deepcopy = deepcopy(batch_t.contiguous().view(-1))
      predictions_deepcopy = deepcopy(predictions.contiguous().view(-1))
      
      #all_ground_truth.detach().cpu().numpy()
      
      batch_accuracy = accuracy_score(batch_t_deepcopy.detach().cpu().numpy(),predictions_deepcopy.detach().cpu().numpy())
      dev_loss+= loss.item()                                                                        
      dev_acc += batch_accuracy

      counter+= 1

    precision,recall,f_score,_ = precision_recall_fscore_support(all_ground_truth.detach().cpu().numpy(), all_predictions.detach().cpu().numpy(), labels = [0,1],average=None)
    scores_dict['precision_0'] = precision[0]
    scores_dict['recall_0'] = recall[0]
    scores_dict['f_score_0'] = f_score[0]
    
    scores_dict['precision_1'] = precision[1]
    scores_dict['recall_1'] = recall[1]
    scores_dict['f_score_1'] = f_score[1]

    
    dev_acc = dev_acc/counter
    dev_loss = dev_loss/counter
    
    dev_precision_0 = scores_dict['precision_0']
    dev_precision_1 = scores_dict['precision_1']
    
    dev_recall_0 = scores_dict['recall_0']
    dev_recall_1 = scores_dict['recall_1']
    
    
    dev_f_score_0 = scores_dict['f_score_0']
    dev_f_score_1 = scores_dict['f_score_1']
    
   
    validation_accuracies.append(dev_acc)
    validation_losses.append(dev_loss)
    validation_scores.append(scores_dict)
    
    print("Validation Accuracy at Epoch " + str(epoch_no) + ": ",dev_acc)
    writer.add_scalar('Validation Accuracy', dev_acc, epoch_no)
    print("Validation Loss at Epoch " + str(epoch_no) + ": ",dev_loss)
    writer.add_scalar('Validation Loss', dev_loss, epoch_no)
    
    print("Validation Precision for class 0 at Epoch " + str(epoch_no) + ":",dev_precision_0)
    print("Validation Recall for class 0 at Epoch " + str(epoch_no) + ":",dev_recall_0)
    print("Validation F-Score for class 0 at Epoch " + str(epoch_no) + ":",dev_f_score_0)
    
    print("Validation Precision for class 1 at Epoch " + str(epoch_no) + ":",dev_precision_1)
    print("Validation Recall for class 1 at Epoch " + str(epoch_no) + ":",dev_recall_1)
    print("Validation F-Score for class 1 at Epoch " + str(epoch_no) + ":",dev_f_score_1)

 
    if dev_loss <= dev_set_loss_min:

      best_model_epoch = epoch_no  
      dev_set_loss_min = dev_loss

    epoch_no = epoch_no + 1
    print(" ")
        
train_model("Bilstm_han_concat",discourse_and_classification_obj,doc_from_word_obj_train,doc_from_word_obj_dev,doc_from_word_obj_test,wordTrainDataloader,wordDevDataloader,wordTestDataloader,writer = SummaryWriter('runs/Metahphor/Bilstm_han_concat'))

  