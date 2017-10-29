import time
from functools import reduce

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from sklearn.utils import resample

from active_learning.oracle import Oracle
from active_learning.utils import transform_to_labeled_feature_vector

import pandas as pd


def label_data(data, gold_standard):
    oracle = Oracle(gold_standard)
    t = time.process_time()
    for record in data:
        record['is_match'] = oracle.is_match(record['abt_record']['record_id'],
                                             record['buy_record']['record_id'])

    number_of_matches = reduce(lambda x, r: x + (1 if r['is_match'] is True else 0), data, 0)

    print('initially labeled all entries ({}) with ground truth. Took {}s and we got {} matches'.format(
        len(data), time.process_time() - t, number_of_matches))


def print_metrics(y_pred, y_test):
    # all those metrics will automatically assume that 1 is the positive class and 0 is the negative one
    # https://goo.gl/FXAS4o
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_metric = f1_score(y_test, y_pred)
    print()
    print('=========== Results ===========')
    print('accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy, precision, recall, f1_metric))
    print('=========== Results ===========')
    print()
    print('----------- Confusion Matrix -----------')
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm))
    print('----------- Confusion Matrix -----------')
    print()
    print('----------- Classification Matrix -----------')
    print(classification_report(y_test, y_pred))
    print('----------- Classification Matrix -----------')
