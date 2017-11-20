from py_stringmatching import JaroWinkler

from preprocessing.word_vector_similarity import WordVectorSimilarity

import multiprocessing as mp


def _calculate_pairs_with_similarities_old_way(pairs_blocked, word_vector_similarities):
    pairs_with_similarities = []
    for idx, pair in enumerate(pairs_blocked):
        pairs_with_similarities.append({
            'abt_record': pair['abt_record'],
            'buy_record': pair['buy_record'],
            # 'edit_similarity': edit_distance(pair['abt_record']['clean_string'], pair['buy_record']['clean_string']),
            # 'tfidf_cosine_similarity': tf_idf_cosine_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
            #                                                                   pair['buy_record']['bag_of_words']),
            # 'monge_elkan_similarity': monge_elkan_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
            #                                                                pair['buy_record']['bag_of_words']),
            # 'generalized_jaccard_similarity': generalized_jaccard_sim.calculate_similarity(
            #     pair['abt_record']['bag_of_words'],
            #     pair['buy_record']['bag_of_words']),
            'word_similarities_vector': word_vector_similarities.get_word_vector_similarities_tf_idf(
                pair['abt_record']['bag_of_words'],
                pair['buy_record']['bag_of_words'])
        })

        if idx % 100 == 0:
            print('Calculated similarities for {} of {} paris'.format(idx, len(pairs_blocked)))
    return pairs_with_similarities


def calculate_similarity(pair, word_vector_similarities):
    print('calculate similarity for pair with ids {}/{}'
          .format(pair['abt_record'].entry_id, pair['buy_record'].entry_id))

    return {
        'abt_record': pair['abt_record'],
        'buy_record': pair['buy_record'],
        # 'edit_similarity': edit_distance(pair['abt_record']['clean_string'], pair['buy_record']['clean_string']),
        # 'tfidf_cosine_similarity': tf_idf_cosine_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
        #                                                                   pair['buy_record']['bag_of_words']),
        # 'monge_elkan_similarity': monge_elkan_sim.calculate_similarity(pair['abt_record']['bag_of_words'],
        #                                                                pair['buy_record']['bag_of_words']),
        # 'generalized_jaccard_similarity': generalized_jaccard_sim.calculate_similarity(
        #     pair['abt_record']['bag_of_words'],
        #     pair['buy_record']['bag_of_words']),
        'word_similarities_vector': word_vector_similarities.get_word_vector_similarities_tf_idf(
            pair['abt_record']['bag_of_words'],
            pair['buy_record']['bag_of_words'])
    }


class SimilarityCalculator:
    def __init__(self):
        print('Initialize an Random forest learner')

        self.prediction = []

    def calculate_pairs_with_similarities(self, pairs_blocked, all_bag_of_words):
        # tf_idf_cosine_sim = SoftTfIdfSimilarity(all_bag_of_words)
        # monge_elkan_sim = MongeElkanSimilarity()
        # generalized_jaccard_sim = GeneralizedJaccardSimilarity()

        word_vector_similarities = WordVectorSimilarity(all_bag_of_words, JaroWinkler().get_raw_score)

        # pairs_with_similarities = _calculate_pairs_with_similarities_old_way(pairs_blocked,
        #                                                                           word_vector_similarities)

        pool = mp.Pool(processes=4)
        pairs_with_similarities = pool.map(lambda pair: calculate_similarity(pair, word_vector_similarities),
                                           pairs_blocked[:1000])

        return pairs_with_similarities
