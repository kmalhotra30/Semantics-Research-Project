import json
import logging
import argparse
import yaml
import torch
import pickle

from data import MetaphorDataset
from train import train
import models
from metrics import *
from evaluate import evaluate

import numpy as np
import torch.nn.functional as F
from collections import defaultdict
import copy
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import pandas as pd


MODELS = ["BL3_LSTM","HAN_LSTM","VA_LSTM"]

def evaluate_analysis(dataset, model, metaphor_metrics, report=True, test=True):
	"""
	Evaluate the model on the given dataloader

	:param dataset:
	:param model:
	:param metaphor_metrics: MultiClassMetric object
	:report: whether to log the f1 scores
	:test: whether the test set is being used
	:return:
	"""
	# Set model to eval mode, which turns off dropout.
	model.eval()
	criterion = torch.nn.NLLLoss(ignore_index=-1)
	subset = dataset.test if test else dataset.validation

	cw = dataset.context_window
	dictionary_pandas = {'focus_word_idxs':[], 'true_labels':[], 'pred_labels':[], 'discourse':[], 'f1':[], 'discourse_length': [], 'focus_length' :[]}

	bin_for_lengths = [[],[],[],[],[],[],[],[],[],[]]
	bin_for_lengths_f1 = [0,0,0,0,0,0,0,0,0,0]

	predictions_for_lengths = [[],[],[],[],[],[],[],[],[],[]]
	labels_for_lengths = [[],[],[],[],[],[],[],[],[],[]]

	
	with torch.no_grad():
		trace = []
		losses = []

		for batch in iter(dataset.dataloader(subset)):


			max_length = batch.idx.size(1)
			bs = int(batch.idx.size(0)/(2*cw+1))

			t = []
			for sent in copy.deepcopy(batch.idx):
				t.append(sent.cpu().detach().numpy())


			discourse_sentences_csv = np.array(t).reshape(bs, 2 * cw + 1, max_length)

			idxs = copy.deepcopy(batch.focus_positions).cpu().detach().numpy()

			idxs = idxs.astype(int)

			focus_sentences_csv = discourse_sentences_csv[range(len(discourse_sentences_csv)),idxs]

			# focus_sentences_csv = discourse_sentences_csv[:, idxs,:]
			
			# focus_sentences_csv = focus_sentences_csv[:, 0,:]

			t1 = copy.deepcopy(batch.lengths).cpu().detach().numpy()

			discourse_lengths_csv = t1.reshape(bs, 2 * cw + 1)

			#focus_lengths_csv = discourse_lengths_csv[:,idxs][:,0]
			focus_lengths_csv = discourse_lengths_csv[range(len(discourse_lengths_csv)),idxs]
			
			predicted = model(batch)

			labels = batch.label_seqs
			
			losses.append(criterion(predicted.view(-1, 2), labels.view(-1)).item())

			#predicted_labels = torch.max(predicted.view(-1, 2),dim = 1)

			_, predicted_labels = torch.max(predicted.data, -1)

			predictions_csv = predicted_labels.cpu().detach().numpy().reshape(bs, max_length)

			labels_csv = labels.view(-1).cpu().detach().numpy().reshape(bs, max_length)


			f1_score_csv = []


			for i in range(bs):

				f1_score_csv.append(f1_score(labels_csv[i][labels_csv[i]!=-1], predictions_csv[i][labels_csv[i]!=-1], average=None, labels = [1])[0])

			for i in range(bs):

				dictionary_pandas['focus_word_idxs'].append(focus_sentences_csv[i])
				dictionary_pandas['true_labels'].append(labels_csv[i])
				dictionary_pandas['pred_labels'].append(predictions_csv[i])
				dictionary_pandas['discourse'].append(discourse_sentences_csv[i])
				dictionary_pandas['f1'].append(f1_score_csv[i])
				dictionary_pandas['discourse_length'].append(discourse_lengths_csv[i])
				dictionary_pandas['focus_length'].append(focus_lengths_csv[i])

				l_for_bin = focus_lengths_csv[i]
				f1_for_bin = f1_score_csv[i]
				#predictions_for_lengths = [[],[],[],[],[],[],[],[],[],[]]
				#labels_for_lengths = [[],[],[],[],[],[],[],[],[],[]]

				if l_for_bin <=5 :
					
					bin_for_lengths[0].append(f1_for_bin)
					predictions_for_lengths[0].extend(predictions_csv[i])
					labels_for_lengths[0].extend(labels_csv[i])

				elif l_for_bin >5 and l_for_bin<=10 :
					bin_for_lengths[1].append(f1_for_bin)
					predictions_for_lengths[1].extend(predictions_csv[i])
					labels_for_lengths[1].extend(labels_csv[i])

				elif l_for_bin >10 and l_for_bin<=15 :
					bin_for_lengths[2].append(f1_for_bin)
					predictions_for_lengths[2].extend(predictions_csv[i])
					labels_for_lengths[2].extend(labels_csv[i])

				elif l_for_bin >15 and l_for_bin<=20 :
					bin_for_lengths[3].append(f1_for_bin)
					predictions_for_lengths[3].extend(predictions_csv[i])
					labels_for_lengths[3].extend(labels_csv[i])

				elif l_for_bin >20 and l_for_bin<=25 :
					bin_for_lengths[4].append(f1_for_bin)
					predictions_for_lengths[4].extend(predictions_csv[i])
					labels_for_lengths[4].extend(labels_csv[i])

				elif l_for_bin >25 and l_for_bin<=30 :
					bin_for_lengths[5].append(f1_for_bin)
					predictions_for_lengths[5].extend(predictions_csv[i])
					labels_for_lengths[5].extend(labels_csv[i])

				elif l_for_bin >30 and l_for_bin<=35 :
					bin_for_lengths[6].append(f1_for_bin)
					predictions_for_lengths[6].extend(predictions_csv[i])
					labels_for_lengths[6].extend(labels_csv[i])

				elif l_for_bin >35 and l_for_bin<=40 :
					bin_for_lengths[7].append(f1_for_bin)
					predictions_for_lengths[7].extend(predictions_csv[i])
					labels_for_lengths[7].extend(labels_csv[i])

				elif l_for_bin >40 and l_for_bin<=45 :
					bin_for_lengths[8].append(f1_for_bin)
					predictions_for_lengths[8].extend(predictions_csv[i])
					labels_for_lengths[8].extend(labels_csv[i])

				elif l_for_bin >45 :
					bin_for_lengths[9].append(f1_for_bin)
					predictions_for_lengths[9].extend(predictions_csv[i])
					labels_for_lengths[9].extend(labels_csv[i])

			
			for pred, target, length in zip(predicted_labels, labels, batch.lengths):

				sentence_trace = []

				for i, j in zip(pred[:length], target[:length]):

					sentence_trace.append((int(i.item()), int(j.item())))

				trace.append(sentence_trace)

	for idx in range(len(bin_for_lengths)):

		bin_for_lengths[idx] = np.array(bin_for_lengths[idx])
		predictions_for_lengths[idx] = np.array(predictions_for_lengths[idx])
		labels_for_lengths[idx] = np.array(labels_for_lengths[idx])
		
		predictions_for_lengths[idx] = predictions_for_lengths[idx][labels_for_lengths[idx]!=-1]
		labels_for_lengths[idx] = labels_for_lengths[idx][labels_for_lengths[idx]!=-1]

		
		#bin_for_lengths_f1[idx] = accuracy_score(labels_for_lengths[idx],predictions_for_lengths[idx])


		bin_for_lengths_f1[idx] = f1_score(labels_for_lengths[idx], predictions_for_lengths[idx], average=None, labels = [1])[0]

		#print(bin_for_lengths_f1[idx])




	#print(bin_for_lengths_f1)

	#lens = [len(b_f_l) for b_f_l in bin_for_lengths]

	#print(lens)


	metaphor_metrics.update(subset, trace, True, np.mean(losses))

	pd.DataFrame(dictionary_pandas).to_csv('xyz.csv')



def main(args, metric, rep):

	filepath = 'model0.model'
	# filepath1 = 'cw_2_model_han0.model'
	# filepath2 = 'cw_2_model_han1.model'
	# filepath3 = 'cw_2_model_han2.model'

	# Data pre-processing
	w2i = json.load(open(args["w2i_file"]))
	i2w = json.load(open(args["i2w_file"]))
	logging.info("Preprocessing data for METAPHOR task.")
	metaphor_dataset = MetaphorDataset(args, w2i=w2i, i2w=i2w, rep=rep)

	# Instantiate the model
	
	# model_choice = args["model_type"]
	# if model_choice == "BL3_LSTM":

 #       model = models.BL3_LSTM(**args)
 #    elif model_choice == "HAN_LSTM":

 #       model = models.HAN_LSTM(**args)  
		
 #    elif model_choice == "VA_LSTM":

 #       model = models.VA_LSTM(**args)

	model_choice = args["model_type"]
	if model_choice == "BL3_LSTM":

		model = models.BL3_LSTM(**args)
	elif model_choice == "HAN_LSTM":

		model = models.HAN_LSTM(**args)

	elif model_choice == "VA_LSTM":

		model = models.VA_LSTM(**args)

	evaluate_analysis(metaphor_dataset, best_model, metric, test=True)
	

if __name__ == "__main__":
	logging.getLogger().setLevel(logging.INFO)
	logging.basicConfig(format='%(asctime)s - %(message)s',
						datefmt='%d-%b %H:%M:%S')
	parser = argparse.ArgumentParser()

	# Task independent arguments related to preprocessing or training
	group = parser.add_argument_group("general")
	group.add_argument("--device", type=int, default=0)
	group.add_argument("--w2i_file", default="embeddings/w2i.json")
	group.add_argument("--i2w_file", default="embeddings/i2w.json")
	group.add_argument("--glove_filename",
					   default="embeddings/embeddings.pickle")
	group.add_argument("--elmo_weights", default="cached_elmo/weights_1024.hdf5")
	group.add_argument("--elmo_options", default="cached_elmo/options_1024.json")
	group.add_argument("--train_steps", type=int, default=10000)
	group.add_argument("--batch_size", type=int, default=64)
	group.add_argument("--repetitions", type=int, default=1)
	group.add_argument("--lr", type=float, default=0.005)
	group.add_argument("--hidden_size", type=int, default=100)
	group.add_argument("--num_layers", type=int, default=1)
	group.add_argument("--dropout1", type=float, default=0.5)
	group.add_argument("--dropout2", type=float, default=0.0)
	group.add_argument("--dropout3", type=float, default=0.1)
	group.add_argument("--output_name", type=str, default="model")
	group.add_argument("--meta_target_type", default="multiclass")
	group.add_argument("--context_window", type=int, default=1)
	group.add_argument("--model_type", type=str, default="BL3_LSTM", choices=MODELS)

	# Arguments related to the metaphor data
	group.add_argument("--meta_folder", default="VUAsequence")
	group.add_argument("--meta_train", default="train.csv")
	group.add_argument("--meta_valid", default="valid.csv")
	group.add_argument("--meta_test", default="test.csv")
	group.add_argument("--sort_data",type = int , default=1)

	args = vars(parser.parse_args())

	if torch.cuda.is_available():
		torch.cuda.set_device(args["device"])

	metric = MultiClassMetric(num_classes=2, task="METAPHOR", dataset="Test")

	# Run multiple times and average the results of the test sets
	for i in range(args["repetitions"]):
		logging.info(f"\n\nRepetition {i}\n-----------------------------------")
		main(args, metric, i)

	# Output statistics over multiple runs for METAPHOR
	logging.info("Final results for the METAPHOR task.")
	metric.report_average()
	metric.write_trace(f"{args['output_name']}.csv")



