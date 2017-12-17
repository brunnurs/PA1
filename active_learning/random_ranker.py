from sklearn.utils import random

from active_learning.ranker import Ranker


class RandomRanker(Ranker):
    def __init__(self):
        print('initialize a rather simple random ranker')

    def rank(self, learner, data, batch_size):
        return random.sample_without_replacement(len(data), batch_size)
