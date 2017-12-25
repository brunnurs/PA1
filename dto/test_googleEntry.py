from unittest import TestCase

from dto.google_entry import GoogleEntry


class TestGoogleEntry(TestCase):
    def test_clean_long_product_numbers(self):
        google_entry = GoogleEntry('http://www.google.com/base/feeds/snippets/6524009149304761284',
                                'onone software inc. pfb-10211 - pro digital frame bundle 1u',
                                'onone software inc. pfb-10211 : what s it like to have jim divitale jack davis '
                                'helene glassman rick sammon and vincent versace at your beck and call? find out for '
                                'yourself with the all new pro digital 10119 : frame bundle. now you too can get the same ...',
                                '',
                                '92.97')
        product_name_without_product_number = ' '.join(google_entry.transform_to_bag_of_words_name())
        self.assertEqual(product_name_without_product_number, 'onone software inc pro digital frame bundle 1u')

        print(' '.join(google_entry.transform_to_bag_of_words()))
