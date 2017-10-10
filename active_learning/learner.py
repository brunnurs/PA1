from abc import ABC, abstractmethod


class Learner(ABC):
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass
