import csv

from dto.abt_entry import AbtEntry
from dto.buy_entry import BuyEntry


class DataImporter:
    ABT_FILE_NAME = 'Abt.csv'
    BUY_FILE_NAME = 'Buy.csv'

    def __init__(self):
        print('hello world')

    def _import_abt(self, folder):
        abt_data = []
        with open("{}/{}".format(folder, self.ABT_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                abt_data.append(AbtEntry(row['id'], row['name'], row['description'], row['price']))

        return abt_data

    def _import_buy(self, folder):
        buy_data = []
        with open("{}/{}".format(folder, self.BUY_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                buy_data.append(BuyEntry(row['id'], row['name'], row['description'], row['manufacturer'], row['price']))

        return buy_data

    def import_abt_buy_dataset(self, folder):
        abt_data = self._import_abt(folder)
        buy_data = self._import_buy(folder)

        return abt_data, buy_data
