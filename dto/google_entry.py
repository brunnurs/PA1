from preprocessing.preprocessing import bag_of_words
import re


# currently not used as it does not improve the results!
def remove_long_product_numbers(value):
    # match every token which has a number of 5 or more digits in it. A token is a string that starts and ends with a space
    pattern = '[^ ]*[0-9]{5}[^ ]*'

    for match in re.finditer(pattern, value):
        print(match.group())
        # print(match.span())

    return re.sub(pattern, '', value)


class GoogleEntry:
    def __init__(self, entry_id, name, description, manufacturer, price):
        self.price = price
        self.manufacturer = manufacturer
        self.description = description
        self.name = name
        self.entry_id = entry_id

    def __str__(self) -> str:
        return 'id:{}, name:{}'.format(self.entry_id, self.name)

    def transform_to_bag_of_words(self):
        # description_pruned = remove_long_product_numbers(self.description)
        return bag_of_words(' '.join([self.description, self.manufacturer]))

    def transform_to_bag_of_words_name(self):
        # name_pruned = remove_long_product_numbers(self.name)
        return bag_of_words(self.name)

    def transform_to_clean_string(self):
        return ' '.join(self.transform_to_bag_of_words())
