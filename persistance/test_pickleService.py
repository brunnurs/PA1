from unittest import TestCase

from persistance.pickle_service import PickleService


class TestPickleService(TestCase):
    def test_save_and_load_preprocessed_data(self):
        # given
        sut = PickleService()

        data = {'some': 'dummy', 'data': 1231444}

        # when
        sut.save_pre_processed_data(data, './')
        reloaded_data = sut.load_pre_processed_data('./')

        # then
        self.assertEqual(reloaded_data, data)
