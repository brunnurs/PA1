from __future__ import division
import collections

from py_stringmatching import utils
from py_stringmatching.similarity_measure.jaro import Jaro

import numpy as np


class WordVectorSimilarity:
    def __init__(self, corpus_list=None, sim_func=Jaro().get_raw_score,
                 threshold=0.5):
        self.__corpus_list = corpus_list
        self.__document_frequency = {}
        self.__compute_document_frequency()
        self.__corpus_size = 0 if self.__corpus_list is None else (
            len(self.__corpus_list))
        self.sim_func = sim_func
        self.threshold = threshold

    def get_word_vector_similarities_simple(self, bag1, bag2):
        # input validations
        utils.sim_check_for_none(bag1, bag2)
        utils.sim_check_for_list_or_set_inputs(bag1, bag2)

        # if the strings match exactly return 1.0
        if utils.sim_check_for_exact_match(bag1, bag2):
            return 1.0

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(bag1, bag2):
            return 0

        # term frequency for input strings
        tf_x, tf_y = collections.Counter(bag1), collections.Counter(bag2)

        # if corpus is not provided treat input string as corpus
        curr_df, corpus_size = (self.__document_frequency, self.__corpus_size)

        # calculating the term sim score against the input string 2,
        # construct similarity map
        similarity_map = {}
        for term_x in tf_x:
            max_score = 0.0
            for term_y in tf_y:
                score = self.sim_func(term_x, term_y)
                # adding sim only if it is above threshold and
                # highest for this element
                if score > self.threshold and score > max_score:
                    similarity_map[term_x] = (term_x, term_y, score)
                    max_score = score

        # position of first string, second string and sim score
        # in the tuple
        first_string_pos = 0
        second_string_pos = 1
        sim_score_pos = 2

        # create a word vector with all the words in the document collection for every comparision.
        # if the word exist in this similarity-map, add the soft TF/ID value. If not, add a 0
        word_similarities_vector = np.zeros(len(curr_df))

        for idx, element in enumerate(curr_df.keys()):
            if element in similarity_map:
                sim = similarity_map[element]
                word_similarities_vector[idx] = sim[sim_score_pos]
            else:
                word_similarities_vector[idx] = 0

        return word_similarities_vector

    def get_word_vector_similarities_tf_idf(self, bag1, bag2):
        # input validations
        utils.sim_check_for_none(bag1, bag2)
        utils.sim_check_for_list_or_set_inputs(bag1, bag2)

        # if the strings match exactly return 1.0
        if utils.sim_check_for_exact_match(bag1, bag2):
            return 1.0

        # if one of the strings is empty return 0
        if utils.sim_check_for_empty(bag1, bag2):
            return 0

        # term frequency for input strings
        tf_x, tf_y = collections.Counter(bag1), collections.Counter(bag2)

        # find unique elements in the input lists and their document frequency
        local_df = {}
        for element in tf_x:
            local_df[element] = local_df.get(element, 0) + 1
        for element in tf_y:
            local_df[element] = local_df.get(element, 0) + 1

        # if corpus is not provided treat input string as corpus
        curr_df, corpus_size = (local_df, 2) if self.__corpus_list is None else (
            (self.__document_frequency, self.__corpus_size))

        # calculating the term sim score against the input string 2,
        # construct similarity map
        similarity_map = {}
        for term_x in tf_x:
            max_score = 0.0
            for term_y in tf_y:
                score = self.sim_func(term_x, term_y)
                # adding sim only if it is above threshold and
                # highest for this element
                if score > self.threshold and score > max_score:
                    similarity_map[term_x] = (term_x, term_y, score)
                    max_score = score

        # position of first string, second string and sim score
        # in the tuple
        first_string_pos = 0
        second_string_pos = 1
        sim_score_pos = 2

        # create a word vector with all the words in the document collection for every comparision.
        # if the word exist in this similarity-map, add the soft TF/ID value. If not, add a 0
        word_similarities_vector = np.zeros(len(curr_df))

        for idx, element in enumerate(curr_df.keys()):
            if element in similarity_map:
                sim = similarity_map[element]
                idf_first = corpus_size / curr_df.get(sim[first_string_pos], 1)
                idf_second = corpus_size / curr_df.get(sim[second_string_pos], 1)
                v_x = idf_first * tf_x.get(sim[first_string_pos], 0)
                v_y = idf_second * tf_y.get(sim[second_string_pos], 0)
                word_similarities_vector[idx] = v_x * v_y * sim[sim_score_pos]
            else:
                word_similarities_vector[idx] = 0

        return word_similarities_vector

    def get_corpus_list(self):
        """Get corpus list.

        Returns:
            corpus list (list of lists).
        """
        return self.__corpus_list

    def get_sim_func(self):
        """Get secondary similarity function.

        Returns:
            secondary similarity function (function).
        """
        return self.sim_func

    def get_threshold(self):
        """Get threshold used for the secondary similarity function.

        Returns:
            threshold (float).
        """
        return self.threshold

    def set_threshold(self, threshold):
        """Set threshold value for the secondary similarity function.

        Args:
            threshold (float): threshold value.
        """
        self.threshold = threshold
        return True

    def set_sim_func(self, sim_func):
        """Set secondary similarity function.

        Args:
            sim_func (function): Secondary similarity function.
        """
        self.sim_func = sim_func
        return True

    def set_corpus_list(self, corpus_list):
        """Set corpus list.

        Args:
            corpus_list (list of lists): Corpus list.
        """
        self.__corpus_list = corpus_list
        self.__document_frequency = {}
        self.__compute_document_frequency()
        self.__corpus_size = 0 if self.__corpus_list is None else (
            len(self.__corpus_list))
        return True

    def __compute_document_frequency(self):
        if self.__corpus_list != None:
            for document in self.__corpus_list:
                for element in set(document):
                    self.__document_frequency[element] = (
                        self.__document_frequency.get(element, 0) + 1)
