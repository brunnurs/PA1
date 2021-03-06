import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from active_learning.metrics import Metrics
from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from passive_learning.sampling import SMOTE_oversampling, ADASYN_oversampling, downsample_to_even_classes, \
    random_oversampling, SMOTEENN_oversampling, random_undersampling
from persistance.pickle_service import PickleService


def explore_random_forest_performance(data, gold_standard):
    """
    The goal of this method is to find out what's the best possible score to get with a random forest model given the
    data we prepare in pre processing.
    """

    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)

    # x, y = downsample_to_even_classes(data)
    # x, y = random_oversampling(data)
    # x, y = ADASYN_oversampling(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    print('train-set shape: {}, {}'.format(np.shape(x_train), np.shape(y_train)))
    print('test-set shape: {}, {}'.format(np.shape(x_test), np.shape(y_test)))

    clf = RandomForestClassifier(n_estimators=500)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    Metrics.print_classification_report_raw(y_pred, y_test)
    # Metrics.plot_precision_recall_curve(y_test, probas_pred)


if __name__ == "__main__":
    print('====== explore random forest performance ======')

    pickle = PickleService()
    ps = pickle.load_pre_processed_data('./data/intermediate_data')
    gs = pickle.load_gold_standard_data('./data/intermediate_data')

    explore_random_forest_performance(ps, gs)
