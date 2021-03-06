from sklearn.ensemble import RandomForestClassifier

from active_learning.learner import Learner
from passive_learning.sampling import SMOTE_oversampling


class RandomForestLearner(Learner):
    def variance(self):
        return 1

    def __init__(self):
        print('Initialize an Random forest learner')
        self.clf = RandomForestClassifier(n_estimators=500, oob_score=True)

        self.prediction = []

    def fit(self, x, y):
        x_sampled, y_sampled = SMOTE_oversampling(x, y)
        self.clf.fit(x_sampled, y_sampled)

    def predict(self, x):
        self.prediction = self.clf.predict(x)
        return self.prediction
