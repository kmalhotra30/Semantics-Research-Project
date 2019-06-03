import logging
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

import torch


def evaluate(dataset, model, metaphor_metrics, report=True, test=False):
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

    with torch.no_grad():
        trace = []
        losses = []

        for batch in iter(dataset.dataloader(subset)):
            predicted = model(batch)
            labels = batch.label_seqs
            losses.append(criterion(predicted.view(-1, 2), labels.view(-1)).item())
            _, predicted_labels = torch.max(predicted.data, -1)

            for pred, target, length in zip(predicted_labels, labels, batch.lengths):
                sentence_trace = []
                for i, j in zip(pred[:length], target[:length]):
                    sentence_trace.append((int(i.item()), int(j.item())))
                trace.append(sentence_trace)

    metaphor_metrics.update(subset, trace, True, np.mean(losses))

