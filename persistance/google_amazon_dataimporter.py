import csv

from dto.abt_entry import AbtEntry
from dto.amazon_entry import AmazonEntry
from dto.buy_entry import BuyEntry
from dto.google_entry import GoogleEntry


class GoogleAmazonDataImporter:
    AMAZON_FILE_NAME = 'Amazon.csv'
    GOOGLE_FILE_NAME = 'GoogleProducts.csv'
    AMAZON_GOOGLE_GOLD_STANDARD_FILE_NAME = 'Amazon_GoogleProducts_perfectMapping.csv'

    def __init__(self):
        print('initialize google/amazon data importer')

    def _import_amazon_products(self, folder):
        amazon_data = {}

        with open("{}/{}".format(folder, self.AMAZON_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entry = AmazonEntry(row['id'], row['title'], row['description'], row['manufacturer'], row['price'])
                amazon_data[entry.entry_id] = entry

        return amazon_data

    def _import_google_products(self, folder):
        google_data = {}

        with open("{}/{}".format(folder, self.GOOGLE_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entry = GoogleEntry(row['id'], row['name'], row['description'], row['manufacturer'], row['price'])
                google_data[entry.entry_id] = entry

        return google_data

    def _import_gold_standard(self, folder, amazon_data, google_data):
        gold_standard = []

        with open("{}/{}".format(folder, self.AMAZON_GOOGLE_GOLD_STANDARD_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    gold_standard.append({
                        'record_a': amazon_data[row['idAmazon']],
                        'record_a_id': row['idAmazon'],
                        'record_b': google_data[row['idGoogleBase']],
                        'record_b_id': row['idGoogleBase']
                    })
                except KeyError:
                    print('could not find objects for row' + row)
        return gold_standard

    def import_amazon_google_dataset(self, folder):
        amazon_data = self._import_amazon_products(folder)
        google_data = self._import_google_products(folder)
        gold_standard = self._import_gold_standard(folder, amazon_data, google_data)

        return amazon_data, google_data, gold_standard
