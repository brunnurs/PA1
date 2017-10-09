from sklearn.svm import SVC


class SvmLearner:
    def __init__(self):
        print('Initialize an SVM learner')
        self.svm_classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

    def fit(self, x, y):
        self.svm_classifier.fit(x, y)

    def predict(self, x):
        return self.svm_classifier.predict(x)