import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from active_learning.metrics import Metrics
from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from persistance.pickle_service import PickleService


def visualize_svm_learner_and_learning_curve(data, gold_standard):
    label_data(data, gold_standard)

    amount_of_data = []
    f1_value_matches = []

    x, y = transform_to_labeled_feature_vector(data)
    for i in range(10, 1000, 10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i, random_state=None,
                                                            stratify=y)

        # clf = SVC(C=100, kernel='linear', class_weight=None, probability=True)
        clf = SVC(C=10, gamma=10, kernel='rbf', class_weight=None, probability=True)
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        Metrics.print_classification_report_raw(y_pred, y_test)
        print('amount of data (diagram and to train the clf): {}'.format(i))

        # plot_clasifier_and_scatter_data(clf, x_train, y_train)

        amount_of_data.append(i)
        f1_value_matches.append(f1_score(y_test, y_pred))

    plt.plot(amount_of_data, f1_value_matches)
    plt.ylabel('f1 value (matches only)')
    plt.xlabel('size training set')
    plt.show()


def plot_clasifier_and_scatter_data(clf, x_train, y_train):
    # plot data points
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30)
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


if __name__ == "__main__":
    print('====== explore linear regression performance ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    visualize_svm_learner_and_learning_curve(ps, gs)
