"""
This file implements metaphor detection for the VUA corpus.

Example Usage
-------------
python3 main.py

Author: Verna Dankers
"""
import json
import logging
import argparse
import yaml
import torch

from data import MetaphorDataset
from train import train
import models
from metrics import *
from evaluate import evaluate


MODELS = ["BL3_LSTM","HAN_LSTM","VA_LSTM"]


def main(args, metric, rep):
    # Data pre-processing
    w2i = json.load(open(args["w2i_file"]))
    i2w = json.load(open(args["i2w_file"]))
    logging.info("Preprocessing data for METAPHOR task.")
    metaphor_dataset = MetaphorDataset(args, w2i=w2i, i2w=i2w, rep=rep)

    # Instantiate the model

    model_choice = args["model_type"]

    if model_choice == "BL3_LSTM":

        model = models.BL3_LSTM(**args)
    elif model_choice == "HAN_LSTM":

        model = models.HAN_LSTM(**args)

    elif model_choice == "VA_LSTM":

        model = models.VA_LSTM(**args)

    best_model = train(model=model, metaphor_dataset=metaphor_dataset, **args)

    # Evaluate metaphor detection
    evaluate(metaphor_dataset, best_model, metric, test=True)
    torch.save(best_model, args["output_name"] + str(rep) + ".model")


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
    group.add_argument("--train_steps", type=int, default=1500)
    group.add_argument("--batch_size", type=int, default=64)
    group.add_argument("--model_type", type=str, default="BL3_LSTM", choices=MODELS)
    group.add_argument("--repetitions", type=int, default=1)
    group.add_argument("--lr", type=float, default=0.005)
    group.add_argument("--hidden_size", type=int, default=100)
    group.add_argument("--num_layers", type=int, default=1)
    group.add_argument("--dropout1", type=float, default=0.5)
    group.add_argument("--dropout2", type=float, default=0.0)
    group.add_argument("--dropout3", type=float, default=0.1)
    group.add_argument("--output_name", type=str, default="model")
    group.add_argument("--meta_target_type", default="multiclass")
    group.add_argument("--context_window", type=int, default=0)
    group.add_argument("--sort_data",type=int, default=1)
    group.add_argument("--use_elmo", type=int, default=0)

    # Arguments related to the metaphor data
    group.add_argument("--meta_folder", default="VUAsequence")
    group.add_argument("--meta_train", default="train.csv")
    group.add_argument("--meta_valid", default="valid.csv")
    group.add_argument("--meta_test", default="test.csv")

    args = vars(parser.parse_args())



    if torch.cuda.is_available():
        torch.cuda.set_device(args["device"])

    metric = MultiClassMetric(num_classes=2, task="METAPHOR", dataset="Test")


    # Run multiple times and average the results of the test sets
    for i in range(args["repetitions"]):
        logging.info('\n\nContext Window : '  + str(args['context_window']))
        logging.info(f"\n\nRepetition {i}\n-----------------------------------")
        main(args, metric, i)

    # Output statistics over multiple runs for METAPHOR
    logging.info("Final results for the METAPHOR task.")
    metric.report_average()
    metric.write_trace(f"{args['output_name']}.csv")
