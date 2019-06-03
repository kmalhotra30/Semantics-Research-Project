"""
This file implements the training algorithm for metaphor detection.

Author: Verna Dankers
"""
import gc
import copy
import os
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from evaluate import evaluate
from metrics import MultiClassMetric


def train(model, metaphor_dataset, train_steps, lr, batch_size, **kwargs):
    """
    Train the model on the different tasks involved.

    Args:
        model (Model): pytorch neural model
        metaphor_dataset (TextDataset): dataloader training set METAPHOR task
        train_steps (int): number of training steps
        lr (float): learning rate
        batch_size (int): size of batches to train on

    Returns:
        Model: best model for METAPHOR task on validation data
    """

    """
    1. Initialize optimizers, criterion, move model to GPU and initialize
    metrics.
    """
    initial_model = copy.deepcopy(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # Use criterion and optimizer
    metaphor_metrics = MultiClassMetric(num_classes=2, task="METAPHOR", dataset="Validation")
    weights = torch.FloatTensor([0.3, 0.7])
    if torch.cuda.is_available():
        weights = weights.cuda()
    criterion = nn.NLLLoss(weight=weights, ignore_index=-1)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Keep track of the best model based on validation data for early stopping
    best_model_meta = None
    best_f1 = 0

    """
    2. Train the model.
    """
    emo_loss, meta_loss, att_loss = [], [], []
    for x in tqdm(range(train_steps)):
        # Avoid memory overflows
        gc.collect()
        torch.cuda.empty_cache()

        # Train mode
        model.train()

        # Collect batch and run one train step
        batch = metaphor_dataset.get_train()
        loss = train_step(batch, optimizer, criterion, model)
        meta_loss.append(loss)

        # Evaluate every 100 steps
        if (x + 1) % 100 == 0:
            logging.info(f"Training step {x+1}")
            """
            3. Evaluate on validation data.
            """
            evaluate(metaphor_dataset, model, metaphor_metrics, test=False)
            if metaphor_metrics.latest > best_f1:
                best_model_meta = copy.deepcopy(model)
                best_f1 = metaphor_metrics.latest
            logging.info(f"METAPHOR - Average training loss: {np.mean(meta_loss):.3f}.")
            meta_loss = []

    logging.info("Finished training.")
    return best_model_meta


def train_step(batch, optimizer, criterion, model):
    predicted = model(batch)
    predicted = predicted.view(-1, 2)
    labels = batch.label_seqs
    labels = labels.view(-1)
    batch_loss = criterion(predicted, labels)
    optimizer.zero_grad()    
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()
