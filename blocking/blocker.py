from py_stringmatching import Jaccard


def has_low_jaccard_similarity(s1_tokenized, s2_tokenized):
    jac = Jaccard()
    return jac.get_sim_score(s1_tokenized, s2_tokenized) < 0.1


