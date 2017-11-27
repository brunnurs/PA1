from py_stringmatching import Jaro

from persistance.dataimporter import DataImporter

if __name__ == "__main__":

    print(Jaro().get_raw_score('lg', 'lg'))
