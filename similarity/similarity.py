from py_stringmatching import Levenshtein, SoftTfIdf, Jaro, MongeElkan, GeneralizedJaccard


def edit_distance(s1, s2):
    lev = Levenshtein()
    return lev.get_sim_score(s1, s2)


class MongeElkanSimilarity:
    def __init__(self) -> None:
        self.monge_elkan = MongeElkan()

    def calculate_similarity(self, s1_tokenized, s2_tokenized):
        return self.monge_elkan.get_raw_score(s1_tokenized, s2_tokenized)


class GeneralizedJaccardSimilarity:
    def __init__(self) -> None:
        self.generalized_jaccard = GeneralizedJaccard()

    def calculate_similarity(self, s1_tokenized, s2_tokenized):
        return self.generalized_jaccard.get_raw_score(s1_tokenized, s2_tokenized)


class SoftTfIdfSimilarity:
    def __init__(self, corpus_list) -> None:
        self.soft_tf_idf = SoftTfIdf(corpus_list, sim_func=Jaro().get_raw_score, threshold=0.8)

    def calculate_similarity(self, s1_tokenized, s2_tokenized):
        """
        See http://anhaidgroup.github.io/py_stringmatching/v0.2.x/SoftTfIdf.html and
        https://en.wikiversity.org/wiki/Duplicate_record_detection#WHIRL for more information about this similarity

        :param s1_tokenized:
        :param s2_tokenized:
        :return:
        """
        return self.soft_tf_idf.get_raw_score(s1_tokenized, s2_tokenized)
