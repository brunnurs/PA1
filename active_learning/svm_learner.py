import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from active_learning.learner import Learner


class SvmLearner(Learner):
    def __init__(self):
        print('Initialize an SVM learner')
        self.clf = SVC(C=10, gamma=10, kernel='rbf', class_weight=None, probability=True)
        # self.clf = SVC(C=100, kernel='linear', class_weight=None, probability=True)

        self.prediction = []

    def fit(self, x, y):
        self.clf.fit(x, y)

        # if len(x) == 20 or len(x) == 60 or len(x) == 300 or len(x) == 600 or len(x) == 990:
        #     self.plot_classifier_and_scatter_data(self.clf, x, y)

    def predict(self, x):
        self.prediction = self.clf.predict(x)
        return self.prediction

    @staticmethod
    def plot_classifier_and_scatter_data(clf, x, y):
        # plot data points
        plt.scatter(x[:, 0], x[:, 1], c=y, s=30)

        # plot the decision function
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = clf.decision_function(xy).reshape(XX.shape)
        # plot decision boundary and margins
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                   linestyles=['--', '-', '--'])
        # plot support vectors
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                   linewidth=1, facecolors='none')

        plt.ylabel('feature #1')
        plt.xlabel('feature #2')

        plt.show()
