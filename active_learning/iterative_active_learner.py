from active_learning.learner import Learner
from active_learning.metrics import Metrics
from active_learning.oracle import Oracle
from active_learning.randome_ranker import RandomRanker
from active_learning.utils import transform_to_labeled_feature_vector, transform_to_feature_vector, \
    map_predictions_to_data, stratified_random_indices


class IterativeActiveLearningAlgorithm:
    def __init__(self, learner: Learner, oracle: Oracle, ranker: RandomRanker, metrics: Metrics, budget: int,
                 batch_size: int, initial_training_data_size: int):
        """
        initialize the active learner. See "Calling Up Crowd-Sourcing to Very Large Datasets:
        A Case for Active Learning" Page 3 for inspiration

        :type initial_labeled_data_percentage: float
        :param budget: amount of interactions with the oracle
        """
        self.metrics = metrics
        self.initial_training_data_size = initial_training_data_size
        self.learner = learner
        self.batch_size = batch_size
        self.budget = budget
        self.ranker = ranker
        self.oracle = oracle

        print('initialize iterative active learning algorithm with budget of {} oracle interactions and a batch-size '
              'of {}. Use {} examples of the data as initial training set'
              .format(budget, batch_size, initial_training_data_size))

    def start_active_learning(self, unlabeled_data):
        self.metrics.label_with_ground_truth(unlabeled_data)

        print('start the iterative learning process')

        initial_training_data_idx = stratified_random_indices(unlabeled_data, self.initial_training_data_size)
        unlabeled_data, labeled_training_data = self._label_training_data(unlabeled_data, initial_training_data_idx)

        self.learner.fit(*transform_to_labeled_feature_vector(labeled_training_data))

        prediction = self.learner.predict(transform_to_feature_vector(unlabeled_data))

        self.metrics.print_statistics(prediction, unlabeled_data, len(labeled_training_data))

        while self.oracle.interactions_with_oracle < self.budget:
            idx_to_ask_oracle = self.ranker.rank(self.learner, unlabeled_data, self.batch_size)

            unlabeled_data, new_labeled_training_data = self._label_training_data(unlabeled_data, idx_to_ask_oracle)
            labeled_training_data.extend(new_labeled_training_data)

            self.learner.fit(*transform_to_labeled_feature_vector(labeled_training_data))
            prediction = self.learner.predict(transform_to_feature_vector(unlabeled_data))

            self.metrics.print_statistics(prediction, unlabeled_data, len(labeled_training_data))

        return map_predictions_to_data(prediction, unlabeled_data)

    def _label_training_data(self, data, idx_list):
        labeled_data = []
        unlabeled_data = []

        for idx, val in enumerate(data):
            if idx not in idx_list:
                unlabeled_data.append(val)
            else:
                val['is_match'] = self.oracle.is_match(val['abt_record']['record_id'],
                                                       val['buy_record']['record_id'])
                labeled_data.append(val)

        return unlabeled_data, labeled_data
