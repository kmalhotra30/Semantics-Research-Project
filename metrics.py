import logging
import math
import scipy
import numpy as np
import torch
import sklearn
import codecs
import torch.nn.functional as F

from collections import defaultdict


class Metric:
    def __init__(self, task, dataset):
        self.task = task
        self.dataset = dataset
        self.traces = []
        self.words = []

    def write_trace(self, filename):
        """
        Open file and write trace, consisting of pairs of booleans.
        :param filename: name of file to write to.
        """
        with codecs.open(filename, 'w', 'utf8') as f:
            for sample, trace in zip(self.words, self.traces):
                if type(trace) == list:
                    prediction, target = zip(*trace)
                    prediction = " ".join([str(x) for x in prediction])
                    target = " ".join([str(x) for x in target])
                f.write(f"{sample.sentence}\t{prediction}\t{target}\n")


class MultiClassMetric(Metric):
    def __init__(self, num_classes, task, dataset):
        super().__init__(task, dataset)
        # Per class measures
        self.precision = defaultdict(list)
        self.recall = defaultdict(list)
        self.f1 = defaultdict(list)
        self.accuracy = []
        self.loss = []
        self.num_classes = num_classes

    def update(self, words, trace, report=False, loss=-1):
        """
        Add metrics for one evaluation round:
        - Precision
        - Recall
        - F1 (macro-averaged)
        - Accuracy (equals micro-averaged F1 in this case)
        :param trace: list of pairs of booleans.
        :param report: whether to log outcome.
        :param loss: loss from evaluation, to report along with metrics.
        """
        if loss != -1:
            self.loss.append(loss)
        self.traces.extend(trace)
        self.words.extend(words)
        accuracy = []
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))

        # i is the prediction, j is the label
        for sentence in trace:
            for i, j in sentence:
                if i == -1 or j == -1:
                    continue
                confusion_matrix[i][j] += 1
                accuracy.append(i == j)
        self.accuracy.append(np.mean(accuracy))

        all_tp, all_fp, all_fn = 0, 0, 0

        # Compute precision, recall and f1 from confusion matrix
        for class_n in range(self.num_classes):
            true_positives = confusion_matrix[class_n, class_n]
            all_tp += true_positives
            false_positives = np.sum(confusion_matrix[class_n, :]) - true_positives
            all_fp += false_positives
            false_negatives = np.sum(confusion_matrix[:, class_n]) - true_positives
            all_fn += false_negatives
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = (2 * precision * recall) / (precision + recall)
            if math.isnan(f1):
                f1 = 0
            if math.isnan(precision):
                precision = 0
            if math.isnan(recall):
                recall = 0

            # Collect metrics per class for future use
            self.precision[class_n].append(precision)
            self.recall[class_n].append(recall)
            self.f1[class_n].append(f1)

        if report:
            self.report_latest(loss)
        self.latest = self.f1[1][-1]

    def report_latest(self, loss):
        """
        Log metrics of last evaluation round.
        :param loss: numerical loss from evaluation function.
        """
        logging.info(
            f"{self.task} - {self.dataset} - Loss {loss:.4f}, Precision {self.precision[1][-1]:.3f}, "+\
            f"Recall {self.recall[1][-1]:.3f}, F_1 {self.f1[1][-1]:.3f}, Accuracy {self.accuracy[-1]:.3f}."
        )

    def report_average(self):
        """
        Log mean metrics per class (precision, recall, f1).
        """
        # Report performance per class
        for class_n in range(self.num_classes):
            p = np.mean(self.precision[class_n])
            p_std = np.std(self.precision[class_n])
            r = np.mean(self.recall[class_n])
            r_std = np.std(self.recall[class_n])
            f1 = np.mean(self.f1[class_n])
            f1_std = np.std(self.f1[class_n])
            logging.info(
                f"{self.task} - {self.dataset} - Class {class_n} - "+\
                f"Precision {p:.3f} +/- {p_std:.3f}, Recall {r:.3f} +/- {r_std:.3f}, "+\
                f"F1 {f1:.3f} +/- {f1_std:.3f}."
            )
        logging.info(f"Accuracy {self.accuracy[-1]:.3f}")

