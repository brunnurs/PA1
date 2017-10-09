import pickle


class PickleService:
    PRE_PROCESSED_DATA_FILE_NAME = 'pre_processed_data.pickle'
    GOLD_STANDARD_DATA_FILE_NAME = 'gold_standard_data.pickle'

    def __init__(self):
        print('initialize pickle service for saving/loading intermediate data')

    def save_pre_processed_data(self, data, folder):
        with open("{}/{}".format(folder, self.PRE_PROCESSED_DATA_FILE_NAME), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pre_processed_data(self, folder):
        with open("{}/{}".format(folder, self.PRE_PROCESSED_DATA_FILE_NAME), 'rb') as handle:
            return pickle.load(handle)

    def save_gold_standard_data(self, data, folder):
        with open("{}/{}".format(folder, self.GOLD_STANDARD_DATA_FILE_NAME), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_gold_standard_data(self, folder):
        with open("{}/{}".format(folder, self.GOLD_STANDARD_DATA_FILE_NAME), 'rb') as handle:
            return pickle.load(handle)
