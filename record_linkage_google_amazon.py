import sys

import numpy

import time

from py_stringmatching import JaroWinkler

from active_learning.ensemble_learner import EnsembleLearner
from active_learning.iterative_active_learner import IterativeActiveLearningAlgorithm
from active_learning.metrics import Metrics
from active_learning.multiple_metrics_plotter import plot_multiple_metrics_learning_curve
from active_learning.oracle import Oracle
from active_learning.random_forest_learner import RandomForestLearner
from active_learning.random_ranker import RandomRanker
from active_learning.svm_learner import SvmLearner
from active_learning.uncertaincy_ranker import UncertaintyRanker
from blocking.blocker import has_low_jaccard_similarity
from persistance.dataimporter import DataImporter
from persistance.google_amazon_dataimporter import GoogleAmazonDataImporter
from persistance.pickle_service import PickleService
from preprocessing.word_vector_similarity import WordVectorSimilarity
from similarity.similarity import edit_distance, SoftTfIdfSimilarity, MongeElkanSimilarity, GeneralizedJaccardSimilarity

import multiprocessing as mp

from similarity.similarity_calculator import SimilarityCalculator


def _pretty_print_order_by_cosine_desc(pairs_with_similarities):
    sorted_desc = sorted(pairs_with_similarities, key=lambda r: r['tfidf_cosine_similarity'], reverse=True)
    print('\r\n'.join(map(
        lambda r: 'cos: {}, edit: {}, s1:{}, s2:{}'.format(r['tfidf_cosine_similarity'], r['edit_similarity'],
                                                           r['abt_record']['clean_string'],
                                                           r['buy_record']['clean_string']), sorted_desc)))


def pre_processing():
    t = time.process_time()
    data_importer = GoogleAmazonDataImporter()

    # import data
    amazon_data, google_data, gold_standard = data_importer.import_amazon_google_dataset('./data/Amazon_GoogleProducts')

    print('====== data import done ======')
    print('abt amazon_data has {} records'.format(len(amazon_data)))
    print('google_data has {} records'.format(len(google_data)))
    print('gold standard has {} true matches'.format(len(gold_standard)))

    # print('\r\n'.join(map(str, gold_standard)))

    # data pre-process data
    for amazon_idx in amazon_data.keys():
        amazon_data[amazon_idx] = {
            'record_id': amazon_data[amazon_idx].entry_id,
            'bag_of_words': amazon_data[amazon_idx].transform_to_bag_of_words(),
            'bag_of_words_name': amazon_data[amazon_idx].transform_to_bag_of_words_name(),
            'clean_string': amazon_data[amazon_idx].transform_to_clean_string(),
        }

    for google_data_key in google_data.keys():
        google_data[google_data_key] = {
            'record_id': google_data[google_data_key].entry_id,
            'bag_of_words': google_data[google_data_key].transform_to_bag_of_words(),
            'bag_of_words_name': google_data[google_data_key].transform_to_bag_of_words_name(),
            'clean_string': google_data[google_data_key].transform_to_clean_string()
        }

    pairs_blocked = []

    print('====== the cartesian product contains {} comparision records ======'
          .format(len(amazon_data) * len(google_data)))

    # build cartesian (n * m) dataset and already calculate simple jaccard similarity
    for amazon_idx, amazon_value in amazon_data.items():
        for google_idx, google_value in google_data.items():

            if not has_low_jaccard_similarity(amazon_value['bag_of_words_name'] + amazon_value['bag_of_words'],
                                              google_value['bag_of_words_name'] + google_value['bag_of_words']):
                pairs_blocked.append({
                    'record_a': amazon_value,
                    'record_b': google_value
                })

    print('====== blocking done. We are dealing now with {} pairs. Took {} s ======'.format(len(pairs_blocked), time.process_time() - t))

    corpus_list_description = []
    corpus_list_name = []
    corpus_list_combined = []

    for pair in pairs_blocked:

        length_a = len(pair['record_a']['bag_of_words'])
        length_b = len(pair['record_b']['bag_of_words'])

        if length_b >= 42 and length_a >= 42:
            pair['record_a']['bag_of_words'] = pair['record_a']['bag_of_words'][:length_b]
            pair['record_b']['bag_of_words'] = pair['record_b']['bag_of_words'][:length_b]

        corpus_list_description.append(pair['record_a']['bag_of_words'])
        corpus_list_description.append(pair['record_b']['bag_of_words'])
        corpus_list_name.append(pair['record_a']['bag_of_words_name'])
        corpus_list_name.append(pair['record_b']['bag_of_words_name'])
        corpus_list_combined.append(pair['record_a']['bag_of_words_name'] + pair['record_a']['bag_of_words'])
        corpus_list_combined.append(pair['record_b']['bag_of_words_name'] + pair['record_b']['bag_of_words'])

    print('====== gathered all bag of words to calculate similarities ======')

    pairs_with_similarities = SimilarityCalculator().calculate_pairs_with_similarities(pairs_blocked,
                                                                                       corpus_list_description,
                                                                                       corpus_list_name,
                                                                                       corpus_list_combined)

    # _pretty_print_order_by_cosine_desc()

    print('====== calculated similarities for all pairs. Took {} s ======'.format(time.process_time() - t))

    return gold_standard, pairs_with_similarities


def active_learning(gold_standard, pairs_with_similarities):

    active_learning_runs = 10

    print('====== start active learning with {} runs ======'.format(active_learning_runs))

    multiple_metrics = []
    for run in range(active_learning_runs):

        metrics_oracle = Oracle(gold_standard)
        metrics = Metrics(metrics_oracle)

        # learner = SvmLearner()
        # learner = RandomForestLearner()
        learner = EnsembleLearner(RandomForestLearner, 8)
        oracle = Oracle(gold_standard)
        # ranker = RandomRanker()
        ranker = UncertaintyRanker()
        budget = 1400
        batch_size = 10
        initial_training_data_size = 1000

        iterative_active_learning = IterativeActiveLearningAlgorithm(learner, oracle, ranker, metrics, budget, batch_size,
                                                                     initial_training_data_size)

        iterative_active_learning.start_active_learning(pairs_with_similarities)

        print('====== run {} done! ======'.format(run))

        # metrics.plot_learning_curve()
        multiple_metrics.append(metrics)

    plot_multiple_metrics_learning_curve(multiple_metrics)
    print('====== active learning done! ======')


if __name__ == "__main__":

    numpy.random.seed(42)

    print('====== start program with parameters ======')
    print('\r\n'.join(map(str, sys.argv)))

    if sys.argv[1] == 'pre_processing':
        print('====== start pre-processing data ======')

        gs, ps = pre_processing()

        pickle = PickleService()
        pickle.save_pre_processed_data(ps, './data/intermediate_data')
        pickle.save_gold_standard_data(gs, './data/intermediate_data')

        print('====== saved pre-processed data and end program ======')

    elif sys.argv[1] == 'active_learning':
        print('====== start active learning with intermediate data ======')

        pickle = PickleService()
        ps = pickle.load_pre_processed_data('./data/intermediate_data')
        gs = pickle.load_gold_standard_data('./data/intermediate_data')

        active_learning(gs, ps)
