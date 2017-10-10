from sklearn.ensemble import RandomForestClassifier

from active_learning.learner import Learner


class RandomForestLearner(Learner):
    def __init__(self):
        print('Initialize an Random forest learner')
        self.rf_classifier = RandomForestClassifier(n_estimators=500, oob_score=True)

        self.prediction = []

    def fit(self, x, y):
        self.rf_classifier.fit(x, y)

    def predict(self, x):
        self.prediction = self.rf_classifier.predict(x)
        return self.prediction
