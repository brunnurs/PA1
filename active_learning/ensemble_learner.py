import functools
import numpy as np
import multiprocessing as mp

from active_learning.learner import Learner


def _fit_single_learner(learner, x, y, idx_first_match):
    bootstrapped_data_idx = np.random.choice(range(len(x)), len(x), replace=True)

    # this is a bit of a hack... but we need at least one sample of every class (so one match) and this might
    # not be the case in the beginning if we start with little initial data (e.g. 10)
    bootstrapped_data_idx[0] = idx_first_match
    learner.fit(x[bootstrapped_data_idx], y[bootstrapped_data_idx])

    return learner


class EnsembleLearner(Learner):
    def __init__(self, learner, number_of_learners, ):
        self.number_of_learners = number_of_learners
        self.learners = []
        self.current_prediction = None

        print('Initialize an ensemble learner ({} learners) with bootstrapping'.format(number_of_learners))

        for k in range(number_of_learners):
            self.learners.append(learner())

    def fit(self, x, y):

        # first [0] is because the where() returns a tuple, second [0] as we are only interested in the first match
        idx_first_match = np.where(y == 1)[0][0]

        self._fit_parallel_implementation(idx_first_match, x, y)

        # self._fit_simple_implementation(idx_first_match, x, y)

    def _fit_simple_implementation(self, idx_first_match, x, y):
        """
        This implementation is faster for little training-data and little number of learners
        :param idx_first_match:
        :param x:
        :param y:
        """
        for clf in self.learners:
            bootstrapped_data_idx = np.random.choice(range(len(x)), len(x), replace=True)

            # this is a bit of a hack... but we need at least one sample of every class (so one match) and this might
            # not be the case in the beginning if we start with little initial data (e.g. 10)
            bootstrapped_data_idx[0] = idx_first_match
            clf.fit(x[bootstrapped_data_idx], y[bootstrapped_data_idx])

    def _fit_parallel_implementation(self, idx_first_match, x, y):
        """
        This implementation is faster for large training-data and larger number of learners
        Try to get a X multiplier of learners where X is the number of logical cores (8 for my notebook)
        :param idx_first_match:
        :param x:
        :param y:
        """
        pool = mp.Pool()
        _internal_fit_function = functools.partial(_fit_single_learner,
                                                   x=x,
                                                   y=y,
                                                   idx_first_match=idx_first_match)

        # it's important to get the learners back, as pool.map() will create copies of the learners
        self.learners = pool.map(_internal_fit_function, self.learners)
        pool.close()  # we are not adding any more processes
        pool.join()  # tell it to wait until all threads are done before going on

    def predict(self, x):
        self.current_prediction = np.asarray([clf.predict(x) for clf in self.learners]).T

        majority = np.apply_along_axis(lambda r: np.argmax(np.bincount(r)), axis=1, arr=self.current_prediction)

        return majority

    def variance(self):

        if self.current_prediction is None:
            raise TypeError('can not call variance without calling predict() first')

        variance = np.apply_along_axis(
            lambda r: (np.count_nonzero(r) / self.number_of_learners) * (
                1 - (np.count_nonzero(r) / self.number_of_learners)),
            axis=1, arr=self.current_prediction)

        return variance
