from functools import reduce

import numpy as np

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    classification_report
from sklearn.svm import SVC

from active_learning.oracle import Oracle
from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from persistance.pickle_service import PickleService

import pandas as pd


def explore_random_forest_performance(data, gold_standard):
    """
    The goal of this method is to find out what's the best possible score to get with a random forest model given the
    data we prepare in pre processing.
    """

    label_data(data, gold_standard)

    # matches = list(filter(lambda r: r['is_match'], data))

    x, y = transform_to_labeled_feature_vector(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    print('============== Tuning hyper-parameters for recall on SVM ==============')

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='recall_macro')
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))  
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()


if __name__ == "__main__":
    print('====== find best SVM by grid search  ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_random_forest_performance(ps, gs)
