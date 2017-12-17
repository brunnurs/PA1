from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import cohen_kappa_score, make_scorer


from active_learning.metrics import Metrics
from active_learning.svm_learner import SvmLearner
from active_learning.utils import transform_to_labeled_feature_vector
from passive_learning.passive_learner_utils import label_data
from passive_learning.sampling import random_oversampling, downsample_to_even_classes, SMOTE_oversampling, \
    ADASYN_oversampling, random_undersampling
from persistance.pickle_service import PickleService


def explore_svm_performance(data, gold_standard):
    """
    The goal of this method is to find out what's the best possible score to get with a random forest model given the
    data we prepare in pre processing.
    """

    label_data(data, gold_standard)

    x, y = transform_to_labeled_feature_vector(data)

    # x, y = downsample_to_even_classes(data)svm
    # x, y = random_oversampling(data)
    # x, y = SMOTE_oversampling(x, y)
    # x, y = ADASYN_oversampling(data)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # x_train, y_train = SMOTE_oversampling(x_train, y_train)

    # find_best_parameters_grid_search(x_test, x_train, y_test, y_train)

    # those parameters have been found by grid search
    # clf = SVC(C=10, gamma=10, kernel='rbf', class_weight={0: 1, 1: 19}, probability=True)
    # clf = SVC(C=10, gamma=10, kernel='rbf', class_weight={0: 1, 1: 19})

    clf = SvmLearner()
    # clf = SVC(C=100, kernel='linear', class_weight=None, probability=True)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    Metrics.print_classification_report_raw(y_pred, y_test)

    probas_pred = clf.clf_.decision_function(x_test)
    Metrics.plot_precision_recall_curve(y_test, probas_pred)


def find_best_parameters_grid_search(x_test, x_train, y_test, y_train):
    print('============== Tuning hyper-parameters for recall on SVM ==============')

    kappa_scorer = make_scorer(cohen_kappa_score)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [10, 1, 1e-1, 1e-2, 1e-3, 1e-4],
                         'C': [1e-1, 1, 1e1, 1e2, 1e3]},
                        {'kernel': ['linear'], 'C': [1e-1, 1, 1e1, 1e2, 1e3]}]
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring=kappa_scorer)
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

    explore_svm_performance(ps, gs)
