from sklearn.utils import random


class RandomRanker:
    def __init__(self):
        print('initialize a rather simple random ranker')

    def rank(self, learner, data, batch_size):
        random.sample_without_replacement(data, batch_size)
