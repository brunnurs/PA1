import sys
from sklearn.utils import random

from active_learning.iterative_active_learner import IterativeActiveLearningAlgorithm
from active_learning.oracle import Oracle
from active_learning.randome_ranker import RandomRanker
from active_learning.svm_learner import SvmLearner
from blocking.blocker import has_low_jaccard_similarity
from persistance.dataimporter import DataImporter
from persistance.pickle_service import PickleService
from similarity.similarity import edit_distance, soft_tf_idf_cosine_similarity


def _pretty_print_order_by_cosine_desc(pairs_with_similarities):
    sorted_desc = sorted(pairs_with_similarities, key=lambda r: r['tfidf_cosine_similarity'], reverse=True)
    print('\r\n'.join(map(
        lambda r: 'cos: {}, edit: {}, s1:{}, s2:{}'.format(r['tfidf_cosine_similarity'], r['edit_similarity'],
                                                           r['abt_record']['clean_string'],
                                                           r['buy_record']['clean_string']), sorted_desc)))


def pre_processing():
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
            'clean_string': abt_data[abt_key].transform_to_clean_string(),
        }

    for buy_data_key in buy_data.keys():
        buy_data[buy_data_key] = {
            'record_id': buy_data[buy_data_key].entry_id,
            'bag_of_words': buy_data[buy_data_key].transform_to_bag_of_words(),
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

    pairs_with_similarities = list(map(lambda r: {
        'abt_record': r['abt_record'],
        'buy_record': r['buy_record'],
        'edit_similarity': edit_distance(r['abt_record']['clean_string'], r['buy_record']['clean_string']),
        'tfidf_cosine_similarity': soft_tf_idf_cosine_similarity(r['abt_record']['bag_of_words'],
                                                                 r['buy_record']['bag_of_words'])}
                                       , pairs_blocked))

    print('====== calculated edit (levenshtein) distance and Soft-TF/IDF cosine similarity for all pairs ======')
    # _pretty_print_order_by_cosine_desc()

    return gold_standard, pairs_with_similarities


def active_learning(gold_standard, pairs_with_similarities):
    learner = SvmLearner()
    oracle = Oracle(gold_standard)
    ranker = RandomRanker()
    budget = 200
    batch_size = 1
    # initial_training_data_percentage = 0.03  # start with 3% of the available gold standard as initial training data
    initial_training_data_percentage = 0.03
    iterative_active_learning = IterativeActiveLearningAlgorithm(learner, oracle, ranker, budget, batch_size,
                                                                 initial_training_data_percentage)
    iterative_active_learning.start_active_learning(pairs_with_similarities)


if __name__ == "__main__":
    print('====== start program with parameters ======')
    print('\r\n'.join(map(str, sys.argv)))

    if sys.argv[1] == 'pre_process_only':
        print('====== start pre-processing data ======')

        gs, ps = pre_processing()

        pickle = PickleService()
        pickle.save_pre_processed_data(ps, './data/intermediate_data')
        pickle.save_gold_standard_data(gs, './data/intermediate_data')

        print('====== saved pre-processed data and end program ======')

    elif sys.argv[1] == 'active_learning_only':
        print('====== start active learning with intermediate data ======')

        pickle = PickleService()
        ps = pickle.load_pre_processed_data('./data/intermediate_data')
        gs = pickle.load_gold_standard_data('./data/intermediate_data')

        active_learning(gs, ps)
    else:
        print('====== start both, pre-processing and active learning ======')

        gs, ps = pre_processing()
        active_learning(gs, ps)


