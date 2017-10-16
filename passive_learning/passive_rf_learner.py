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

    print('train-set shape: {}, {}'.format(np.shape(x_train), np.shape(y_train)))
    print('test-set shape: {}, {}'.format(np.shape(x_test), np.shape(y_test)))

    clf = RandomForestClassifier(n_estimators=500, oob_score=True)

    clf.fit(x_train, y_train)

    print('Random forest score: {}'.format(clf.score(x_test, y_test)))

    y_predicted = clf.predict(x_test)

    # all those metrics will automatically assume that 1 is the positive class and 0 is the negative one
    # https://goo.gl/FXAS4o
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1_metric = f1_score(y_test, y_predicted)

    print('accuracy: {}, precision: {}, recall: {}, f1: {}'.format(accuracy, precision, recall, f1_metric))

    cm = confusion_matrix(y_test, y_predicted)

    print(pd.DataFrame(cm))

    print(classification_report(y_test, y_predicted))


if __name__ == "__main__":
    print('====== explore random forest performance ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_random_forest_performance(ps, gs)
