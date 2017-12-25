import time
from functools import reduce

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report

from active_learning.oracle import Oracle


def label_data(data, gold_standard):
    oracle = Oracle(gold_standard)
    t = time.process_time()
    for record in data:
        record['is_match'] = oracle.is_match(record['record_a']['record_id'],
                                             record['record_b']['record_id'])

    number_of_matches = reduce(lambda x, r: x + (1 if r['is_match'] is True else 0), data, 0)

    print('initially labeled all entries ({}) with ground truth. Took {}s and we got {} matches'.format(
        len(data), time.process_time() - t, number_of_matches))