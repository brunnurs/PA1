from sklearn.model_selection import train_test_split

import numpy

from active_learning.ensemble_learner import EnsembleLearner
from active_learning.metrics import Metrics
from active_learning.svm_learner import SvmLearner
from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from persistance.pickle_service import PickleService


def explore_ensemble_performance(data, gold_standard):

    numpy.random.seed(42)

    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    clf = EnsembleLearner(SvmLearner, 24)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    variance = clf.variance()

    Metrics.print_classification_report_raw(y_pred, y_test)


if __name__ == "__main__":

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_ensemble_performance(ps, gs)
