from unittest import TestCase

from dto.buy_entry import concat_dasherized_expressions


class TestConcat_dasherized_expressions(TestCase):
    def test_concat_dasherized_expressions_1(self):

        with_dash = 'Logitech Z-2300 Multimedia Speaker System - 970118-0403'
        without_dash = 'Logitech Z2300 Multimedia Speaker System - 9701180403'

        self.assertEqual(without_dash, concat_dasherized_expressions(with_dash))

    def test_concat_dasherized_expressions_2(self):

        with_dash = 'Sony ICF-CD73W AM/FM/Weather Shower CD Clock Radio (White)'
        without_dash = 'Sony ICFCD73W AM/FM/Weather Shower CD Clock Radio (White)'

        self.assertEqual(without_dash, concat_dasherized_expressions(with_dash))

    def test_concat_dasherized_expressions_3(self):

        with_dash = 'Panasonic Kx-tg9391t Dect 6.0 Two-line Corded/cordless Phone Combo - KX-TG9391T'
        without_dash = 'Panasonic Kxtg9391t Dect 6.0 Two-line Corded/cordless Phone Combo - KXTG9391T'

        self.assertEqual(without_dash, concat_dasherized_expressions(with_dash))
