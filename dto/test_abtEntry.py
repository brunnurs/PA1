from unittest import TestCase

from dto.abt_entry import AbtEntry


class TestAbtEntry(TestCase):
    def test_transform_to_clean_string(self):
        abt_entry = AbtEntry(9312, 'Panasonic Hands-Free Headset - KXTCA86', 'Panasonic Hands-Free Headset - KXTCA86/ '
                                                                             'Comfort Fit And  Fold Design/ Noise '
                                                                             'Cancelling Microphone/ Standard 2.5mm '
                                                                             'Connection', '$14.95')
        clean_string = abt_entry.transform_to_clean_string()
        print(clean_string)
