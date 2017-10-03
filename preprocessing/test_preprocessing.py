from unittest import TestCase

from preprocessing.preprocessing import bag_of_words


class TestPreprocessing(TestCase):
    def test_clean_word_set(self):
        original_input = "Panasonic 2-Line Integrated Telephone - KXTSC14W/ Call Waiting/ 50-Station Caller ID/ Voice " \
                         "Mail Message-Waiting Indicator/ Speakerphone/ 3-Line LCD Display/ White Finish "
        print(original_input)
        as_word_bag = bag_of_words(original_input)
        print(as_word_bag)