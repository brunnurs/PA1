import heapq

from active_learning.ranker import Ranker


class UncertaintyRanker(Ranker):
    def __init__(self):
        print('initialize Uncertainty ranker')

    def rank(self, learner, data, batch_size):
        variance = learner.variance()

        # nlargest() returns the n largest elements. We wan't not only the value but a (key, value) tuple. To tell
        # nlargest() where the value is, we use provide a proper (and simple) key-function
        largest_variances = heapq.nlargest(batch_size, enumerate(variance), key=lambda key_value: key_value[1])

        # largest_variances is a list of tuples (idx, value). We only want the indices
        # [(87, 0.25), (179, 0.25), (421, 0.25)...]
        return [idx for idx, val in largest_variances]
