from importer.dataimporter import DataImporter

if __name__ == "__main__":
    data_importer = DataImporter()
    abt_data, buy_data = data_importer.import_abt_buy_dataset('./data/ABT_BUY')
    print(*abt_data, sep='\n')


