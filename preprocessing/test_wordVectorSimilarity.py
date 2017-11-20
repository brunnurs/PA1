from unittest import TestCase

from py_stringmatching import SoftTfIdf, Jaro, JaroWinkler

from preprocessing.preprocessing import bag_of_words
from preprocessing.word_vector_similarity import WordVectorSimilarity


class TestWordVectorSimilarity(TestCase):

    s1_simple = 'Ursin Brunner Tester Nonexisting'
    s2_simple = 'Ursin Brunerx Test Randomword Second'
    s3_simple = 'Peter Brunner Homeworld'

    s1_complex = 'Linksys EtherFast 8-Port 10/100 Switch - EZXS88W,Linksys EtherFast 8-Port 10/100 Switch - EZXS88W/ 10/100 ' \
         'Dual-Speed Per-Port/ Perfect For Optimizing 10BaseT And 100BaseTX Hardware On The Same Network/ Speeds Of ' \
         'Up To 200Mbps In Full Duplex Operation/ Eliminate Bandwidth Constraints And Clear Up Bottlenecks '

    s2_complex = 'Linksys EtherFast EZXS88W Ethernet Switch - EZXS88W,Linksys EtherFast 8-Port 10/100 Switch (New/Workgroup),' \
         'LINKSYS '

    def test_get_word_vector_similarities(self):
        all_bag_of_words = []

        s1_tokenized = bag_of_words(self.s1_simple)
        s2_tokenized = bag_of_words(self.s2_simple)
        s3_tokenized = bag_of_words(self.s3_simple)

        all_bag_of_words.append(s1_tokenized)
        all_bag_of_words.append(s2_tokenized)
        all_bag_of_words.append(s3_tokenized)

        sim_engine = WordVectorSimilarity(all_bag_of_words, sim_func=JaroWinkler().get_raw_score, threshold=0.8)

        word_similarities_vector = sim_engine.get_word_vector_similarities_simple(s1_tokenized, s2_tokenized)

        print(word_similarities_vector)

    def test_get_word_vector_similarities_tf_idf_simple_examples(self):
        all_bag_of_words = []

        s1_tokenized = bag_of_words(self.s1_simple)
        s2_tokenized = bag_of_words(self.s2_simple)
        s3_tokenized = bag_of_words(self.s3_simple)

        all_bag_of_words.append(s1_tokenized)
        all_bag_of_words.append(s2_tokenized)
        all_bag_of_words.append(s3_tokenized)

        sim_engine = WordVectorSimilarity(all_bag_of_words, sim_func=Jaro().get_raw_score, threshold=0.8)

        word_similarities_vector = sim_engine.get_word_vector_similarities_simple(s1_tokenized, s2_tokenized)

        print(word_similarities_vector)

        self.assertEqual(len(word_similarities_vector), 10)

    def test_get_word_vector_similarities_tf_idf(self):
        all_bag_of_words = []

        s1_tokenized = bag_of_words(self.s1_complex)
        s2_tokenized = bag_of_words(self.s2_complex)

        all_bag_of_words.append(s1_tokenized)
        all_bag_of_words.append(s2_tokenized)

        sim_engine = WordVectorSimilarity(all_bag_of_words, sim_func=Jaro().get_raw_score, threshold=0.8)

        word_similarities_vector = sim_engine.get_word_vector_similarities_tf_idf(s1_tokenized, s2_tokenized)

        print(word_similarities_vector)

        self.assertEqual(len(word_similarities_vector), 8)