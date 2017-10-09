from active_learning.oracle import Oracle
from active_learning.randome_ranker import RandomRanker
from active_learning.svm_learner import SvmLearner


class IterativeActiveLearningAlgorithm:
    def __init__(self, learner: SvmLearner, oracle: Oracle, ranker: RandomRanker, budget: int, batch_size: int,
                 initial_labeled_data_percentage: float):
        """
        initialize the active learner. See "Calling Up Crowd-Sourcing to Very Large Datasets:
        A Case for Active Learning" Page 3 for inspiration

        :type initial_labeled_data_percentage: float
        :param budget: amount of interactions with the oracle
        """
        self.initial_labeled_data_percentage = initial_labeled_data_percentage
        self.learner = learner
        self.batch_size = batch_size
        self.budget = budget
        self.ranker = ranker
        self.oracle = oracle

        print('initialize iterative active learning algorithm with budget of {} oracle interactions and a batch-size '
              'of {}. Use {}% of the data as initial training set'
              .format(budget, batch_size, initial_labeled_data_percentage))

    def start_active_learning(self, data):
        print('start the iterative learning process')

        initial_labeled_data = self.label_initial_training_data(data)

        # the *some_tuple_value expands the tuple into an argument list
        # https://stackoverflow.com/questions/1993727/expanding-tuples-into-arguments
        self.learner.fit(*self._transform_data_to_feature_vector(initial_labeled_data))
        print('use the first {} records of the data as initial labeled training data and fit the model to it'
              .format(len(initial_labeled_data)))

        return []

    def _transform_data_to_feature_vector(self, labeled_data):
        x = list(map(lambda r: [r['edit_similarity'], r['tfidf_cosine_similarity']], labeled_data))
        y = list(map(lambda r: int(r['is_match']), labeled_data))
        return x, y

    def label_initial_training_data(self, data):
        initial_labeled_data = data[0:int(len(data) * self.initial_labeled_data_percentage)]

        for record in initial_labeled_data:
            record['is_match'] = self.oracle.is_match(record['abt_record']['record_id'],
                                                      record['buy_record']['record_id'])

        # we reset the oracle as we don't count this labels
        self.oracle.reset_interactions()

        return initial_labeled_data
