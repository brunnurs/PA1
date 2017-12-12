import sys

import numpy

import time

from py_stringmatching import JaroWinkler

from active_learning.iterative_active_learner import IterativeActiveLearningAlgorithm
from active_learning.metrics import Metrics
from active_learning.oracle import Oracle
from active_learning.random_forest_learner import RandomForestLearner
from active_learning.randome_ranker import RandomRanker
from active_learning.svm_learner import SvmLearner
from blocking.blocker import has_low_jaccard_similarity
from persistance.dataimporter import DataImporter
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
    data_importer = DataImporter()

    # import data
    abt_data, buy_data, gold_standard = data_importer.import_abt_buy_dataset('./data/ABT_BUY')

    print('====== data import done ======')
    print('abt dataset has {} records'.format(len(abt_data)))
    print('buy dataset has {} records'.format(len(buy_data)))
    print('gold standard has {} true matches'.format(len(gold_standard)))

    # print('\r\n'.join(map(str, gold_standard)))

    # data pre-process data
    for abt_key in abt_data.keys():
        abt_data[abt_key] = {
            'record_id': abt_data[abt_key].entry_id,
            'bag_of_words': abt_data[abt_key].transform_to_bag_of_words(),
            'bag_of_words_name': abt_data[abt_key].transform_to_bag_of_words_name(),
            'clean_string': abt_data[abt_key].transform_to_clean_string(),
        }

    for buy_data_key in buy_data.keys():
        buy_data[buy_data_key] = {
            'record_id': buy_data[buy_data_key].entry_id,
            'bag_of_words': buy_data[buy_data_key].transform_to_bag_of_words(),
            'bag_of_words_name': buy_data[buy_data_key].transform_to_bag_of_words_name(),
            'clean_string': buy_data[buy_data_key].transform_to_clean_string()
        }

    print('====== data pre-processing done ======')

    all_pairs = []

    # build cartesian (n * m) dataset and already calculate simple jaccard similarity
    for abt_key, abt_value in abt_data.items():
        for buy_key, buy_value in buy_data.items():
            all_pairs.append({
                'abt_record': abt_value,
                'buy_record': buy_value,
                'has_low_similarity': has_low_jaccard_similarity(abt_value['bag_of_words'], buy_value['bag_of_words'])})

    print('====== built all the {} comparision records ======'.format(len(all_pairs)))

    pairs_blocked = list(filter(lambda r: not r['has_low_similarity'], all_pairs))

    print('====== blocking done. We are dealing now with {} pairs ======'.format(len(pairs_blocked)))

    corpus_list_original = []
    corpus_list_abt_name_only = []

    for pair in pairs_blocked:
        corpus_list_original.append(pair['abt_record']['bag_of_words'])
        corpus_list_original.append(pair['buy_record']['bag_of_words'])
        corpus_list_abt_name_only.append(pair['abt_record']['bag_of_words_name'])
        corpus_list_abt_name_only.append(pair['buy_record']['bag_of_words'])

    print('====== gathered all bag of words to calculate similarities ======')

    pairs_with_similarities = SimilarityCalculator().calculate_pairs_with_similarities(pairs_blocked,
                                                                                       corpus_list_original,
                                                                                       corpus_list_abt_name_only)

    print('====== calculated similarities for all pairs ======')
    # _pretty_print_order_by_cosine_desc()

    print('====== pre-processing done. Took {} s ======'.format(time.process_time() - t))

    return gold_standard, pairs_with_similarities


def active_learning(gold_standard, pairs_with_similarities):
    metrics_oracle = Oracle(gold_standard)
    metrics = Metrics(metrics_oracle)

    learner = SvmLearner()
    # learner = RandomForestLearner()
    oracle = Oracle(gold_standard)
    ranker = RandomRanker()
    budget = 18000
    batch_size = 10
    initial_training_data_size = 10

    iterative_active_learning = IterativeActiveLearningAlgorithm(learner, oracle, ranker, metrics, budget, batch_size,
                                                                 initial_training_data_size)

    iterative_active_learning.start_active_learning(pairs_with_similarities)

    metrics.plot_learning_curve()

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
