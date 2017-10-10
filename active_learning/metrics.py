import time
from functools import reduce

from sklearn.metrics import classification_report

from active_learning.oracle import Oracle


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

    def print_classification_report(self, predictions, data):
        y_true = list(map(lambda r: int(r['ground_truth']), data))

        print('====== another active learning iteration done. We predict {} records ======'.format(len(data)))
        print(classification_report(y_true=y_true, y_pred=predictions, target_names=['no match', 'match']))
