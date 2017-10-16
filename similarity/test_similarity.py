from unittest import TestCase

from py_stringmatching import Levenshtein, JaroWinkler, Jaro, SoftTfIdf

from preprocessing.preprocessing import bag_of_words


class TestSimilarity(TestCase):
    def test_several_string_distances(self):


        real_value = "Michael Beat Stolz"
        real_world_input_string = "Michael Test Stolz"
        common_mistakes_1 = "Michael Beat Stoltz"
        common_mistakes_2 = "Micheal Beat Stolz"
        common_mistakes_3 = "Michel Beat Stoltz"
        common_mistakes_4 = "Beat Michael Stolz"

        all_mistakes = [real_world_input_string, common_mistakes_1, common_mistakes_2, common_mistakes_3, common_mistakes_4]


        # given
        lev = Levenshtein()
        jw = JaroWinkler()
        jaro = Jaro()
        soft_tf_idf = SoftTfIdf(None, sim_func=Jaro().get_raw_score, threshold=0.8)

        for idx, mistake in enumerate(all_mistakes):
            print("====================Mistake Nr {} ======================".format(idx))
            print("Levenshtein (Edit Distance) {}".format(lev.get_sim_score(mistake, real_value)))
            print("JaroWinkler {}".format(jw.get_sim_score(mistake, real_value)))
            print("Jaro {}".format(jaro.get_sim_score(mistake, real_value)))
            print("SoftTfIdf Cosine {}".format(soft_tf_idf.get_raw_score(bag_of_words(mistake), bag_of_words(real_value))))
            print("==========================================")
