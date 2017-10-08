import csv

from dto.abt_buy_linkage import AbtBuyLinkage
from dto.abt_entry import AbtEntry
from dto.buy_entry import BuyEntry


class DataImporter:
    # ABT_FILE_NAME = 'Abt.csv'
    # BUY_FILE_NAME = 'Buy.csv'
    # ABT_BUY_GOLD_STANDARD_FILE_NAME = 'abt_buy_perfectMapping.csv'
    ABT_FILE_NAME = 'abt_small.csv'  # TODO: activate the real data!
    BUY_FILE_NAME = 'buy_small.csv'
    ABT_BUY_GOLD_STANDARD_FILE_NAME = 'abt_buy_empty.csv'

    def __init__(self):
        print('hello world')

    def _import_abt(self, folder):
        abt_data = {}

        with open("{}/{}".format(folder, self.ABT_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entry = AbtEntry(row['id'], row['name'], row['description'], row['price'])
                abt_data[entry.entry_id] = entry

        return abt_data

    def _import_buy(self, folder):
        buy_data = {}

        with open("{}/{}".format(folder, self.BUY_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                entry = BuyEntry(row['id'], row['name'], row['description'], row['manufacturer'], row['price'])
                buy_data[entry.entry_id] = entry

        return buy_data

    def _import_gold_standard(self, folder, abt_data, buy_data):
        gold_standard = []

        with open("{}/{}".format(folder, self.ABT_BUY_GOLD_STANDARD_FILE_NAME)) as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                try:
                    abt_entry = abt_data[row['idAbt']]
                    buy_entry = buy_data[row['idBuy']]

                    linkage = AbtBuyLinkage(abt_entry, buy_entry)

                    gold_standard.append(linkage)
                except KeyError:
                    print('could not find objects for row' + row)
        return gold_standard

    def import_abt_buy_dataset(self, folder):
        abt_data = self._import_abt(folder)
        buy_data = self._import_buy(folder)
        gold_standard = self._import_gold_standard(folder, abt_data, buy_data)

        return abt_data, buy_data, gold_standard
