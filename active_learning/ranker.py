from abc import ABC, abstractmethod


class Ranker(ABC):
    @abstractmethod
    def rank(self, learner, data, batch_size):
        pass
