import functools
from py_stringmatching import Jaro

from preprocessing.word_vector_similarity import WordVectorSimilarity

import multiprocessing as mp

from similarity.similarity import SoftTfIdfSimilarity

# EVIL global variable :-/ But has just no priority right now...
# https://stackoverflow.com/questions/2080660/python-multiprocessing-and-a-shared-counter
counter = None


# def _calculate_pairs_with_similarities_old_way(pairs_blocked, word_vector_similarities):
#     pairs_with_similarities = []
#     for idx, pair in enumerate(pairs_blocked):
#         pairs_with_similarities.append({
#             'abt_record': pair['abt_record'],
#             'buy_record': pair['buy_record'],
#             # 'tfidf_cosine_similarity': tf_idf_cosine_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
#             #                                                                   pair['buy_record']['bag_of_words']),
#             # 'monge_elkan_similarity': monge_elkan_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
#             #                                                                pair['buy_record']['bag_of_words']),
#             # 'generalized_jaccard_similarity': generalized_jaccard_sim.calculate_similarity(
#             #     pair['abt_record']['bag_of_words'],
#             #     pair['buy_record']['bag_of_words']),
#             'word_similarities_vector': word_vector_similarities.get_word_vector_similarities_tf_idf(
#                 pair['abt_record']['bag_of_words'],
#                 pair['buy_record']['bag_of_words'])
#         })
#
#         if idx % 100 == 0:
#             print('Calculated similarities for {} of {} pairs'.format(idx, len(pairs_blocked)))
#     return pairs_with_similarities


def calculate_similarity(pair, word_vector_similarities, tf_idf_cosine_sim, tf_idf_cosine_sim_abt_name_only):
    global counter
    counter.increment()

    if counter.value % 100 == 0:
        print('Calculated similarities for {} pairs'.format(counter.value))

    return {
        'abt_record': pair['abt_record'],
        'buy_record': pair['buy_record'],

        'tfidf_cosine_similarity': tf_idf_cosine_sim.calculate_similarity(
            pair['abt_record']['bag_of_words'],
            pair['buy_record']['bag_of_words']),

        'tfidf_cosine_similarity_abt_name_only': tf_idf_cosine_sim_abt_name_only.calculate_similarity(
            pair['abt_record']['bag_of_words_name'],
            pair['buy_record']['bag_of_words']),

        'word_similarities_vector': word_vector_similarities.get_word_vector_similarities_tf_idf(
            pair['abt_record']['bag_of_words'],
            pair['buy_record']['bag_of_words'])

        # 'monge_elkan_similarity': monge_elkan_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
        #                                                                pair['buy_record']['bag_of_words']),
        # 'generalized_jaccard_similarity': generalized_jaccard_sim.calculate_similarity(
        #     pair['abt_record']['bag_of_words'],
        #     pair['buy_record']['bag_of_words']),
    }


class SimilarityCalculator:
    def calculate_pairs_with_similarities(self, pairs_blocked, corpus_list_original, corpus_list_abt_name_only):
        # monge_elkan_sim = MongeElkanSimilarity()
        # generalized_jaccard_sim = GeneralizedJaccardSimilarity()

        # the threshold 0.95 is very important! It has been proven by several experiments. Lower values will make it
        # impossible to get good results (reason are most probably product identifier which are very similar, so lower
        # tolerance kill a lot of information)
        tf_idf_cosine_sim = SoftTfIdfSimilarity(corpus_list_original, threshold=0.95)
        tf_idf_cosine_sim_abt_name_only = SoftTfIdfSimilarity(corpus_list_abt_name_only, threshold=0.95)

        word_vector_similarities = WordVectorSimilarity(corpus_list_original, Jaro().get_raw_score, threshold=0.95)

        counter = Counter()
        pool = mp.Pool(processes=6, initializer=counter_initializer, initargs=(counter,))

        # This is only necessary because the pool.map() can not handle lambdas (see
        # https://stackoverflow.com/questions/4827432/how-to-let-pool-map-take-a-lambda-function)
        #
        # pairs_with_similarities = pool.map(lambda pair: calculate_similarity(pair, word_vector_similarities),
        #                                    pairs_blocked)
        similarity_func_with_one_argument = functools.partial(calculate_similarity,
                                                              word_vector_similarities=word_vector_similarities,
                                                              tf_idf_cosine_sim=tf_idf_cosine_sim,
                                                              tf_idf_cosine_sim_abt_name_only=tf_idf_cosine_sim_abt_name_only)

        pairs_with_similarities = pool.map(similarity_func_with_one_argument,
                                           pairs_blocked)

        # pairs_with_similarities2 = _calculate_pairs_with_similarities_old_way(pairs_blocked,
        #                                                                       word_vector_similarities)

        return pairs_with_similarities


def counter_initializer(args):
    global counter
    counter = args


class Counter(object):
    """ Thread-safe counter
    See
    https://stackoverflow.com/questions/2080660/python-multiprocessing-and-a-shared-counter
    """

    def __init__(self):
        self.val = mp.Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value
