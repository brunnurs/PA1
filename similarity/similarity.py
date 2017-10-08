from py_stringmatching import Levenshtein, SoftTfIdf, Jaro


def edit_distance(s1, s2):
    lev = Levenshtein()
    return lev.get_sim_score(s1, s2)


def soft_tf_idf_cosine_similarity(s1_tokenized, s2_tokenized):

    """
    Note: we currently calculate the tf/idf without a corpus list (which would have to be built over
    all data records!). It will be built on the fly over the two input sets. Keep that in mind for improvement

    See http://anhaidgroup.github.io/py_stringmatching/v0.2.x/SoftTfIdf.html and
    https://en.wikiversity.org/wiki/Duplicate_record_detection#WHIRL for more information about this similarity

    :param s1_tokenized:
    :param s2_tokenized:
    :return:
    """
    soft_tf_idf = SoftTfIdf(None, sim_func=Jaro().get_raw_score, threshold=0.8)
    return soft_tf_idf.get_raw_score(s1_tokenized, s2_tokenized)
