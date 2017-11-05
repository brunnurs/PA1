import time
from functools import reduce

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, cohen_kappa_score

from active_learning.oracle import Oracle

import pandas as pd


class Metrics:
    def __init__(self, oracle: Oracle):
        self.oracle = oracle

    def label_with_ground_truth(self, data):
        t = time.process_time()
        for record in data:
            record['ground_truth'] = self.oracle.is_match(record['abt_record']['record_id'],
                                                          record['buy_record']['record_id'])

        number_of_matches = reduce(lambda x, r: x + (1 if r['ground_truth'] is True else 0), data, 0)

        print('initially labeled all entries with ground truth (to get faster metrics). Took {}s and we got {} matches'.format(time.process_time() - t, number_of_matches))

    @staticmethod
    def print_classification_report(y_pred, data):
        y_true = list(map(lambda r: int(r['ground_truth']), data))

        print('********** another active learning iteration done. We predict {} records ********** '.format(len(data)))
        Metrics.print_classification_report_raw(y_pred, y_true)

    @staticmethod
    def print_classification_report_raw(y_pred, y_true):

        # all those metrics will automatically assume that 1 is the positive class and 0 is the negative one
        # https://goo.gl/FXAS4o
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1_metric = f1_score(y_true, y_pred)

        # The Kappa or Cohen’s kappa is the classification accuracy normalized by the imbalance of the classes in the data.
        cohen_kappa = cohen_kappa_score(y_true, y_pred)

        print()
        print('=========== Results ===========')
        print('accuracy: {}, Cohen\'s kappa: {}, precision: {}, recall: {}, f1: {}'
              .format(accuracy, cohen_kappa, precision, recall, f1_metric))
        print('=========== Results ===========')
        print()
        print('----------- Confusion Matrix -----------')
        cm = confusion_matrix(y_true, y_pred)
        print(pd.DataFrame(cm))
        print('----------- Confusion Matrix -----------')
        print()
        print('----------- Classification Matrix -----------')
        print(classification_report(y_true, y_pred, target_names=['no match', 'match']))
        print('----------- Classification Matrix -----------')
        print()
        print()
        print()
