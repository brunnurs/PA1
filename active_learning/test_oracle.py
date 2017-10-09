from unittest import TestCase

from active_learning.oracle import Oracle
from dto.abt_entry import AbtEntry
from dto.buy_entry import BuyEntry


class TestOracle(TestCase):
    def test_is_match(self):
        # given
        gold_standard = [{
            'abt_record': AbtEntry(None, None, None, None),
            'abt_record_id': 6726,
            'buy_record': BuyEntry(None, None, None, None, None),
            'buy_record_id': 10388980
        }, {
            'abt_record': AbtEntry(None, None, None, None),
            'abt_record_id': 3211,
            'buy_record': BuyEntry(None, None, None, None, None),
            'buy_record_id': 99999999
        }]

        sut = Oracle(gold_standard)

        # when
        is_match = sut.is_match(6726, 10388980)

        # then
        self.assertTrue(is_match)
        self.assertEqual(sut.interactions_with_oracle, 1)
