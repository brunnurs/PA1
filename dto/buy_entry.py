from preprocessing.preprocessing import bag_of_words


def concat_dasherized_expressions(value):
    """
    remove dashes if they are in between of a word
    :param value:
    :return:
    """
    dashes_to_remove = []
    for idx, char in enumerate(value):
        if char == '-':
            if value[idx - 1].isalnum() and value[idx + 1].isalnum():
                dashes_to_remove.append(idx)

    pruned_string = value

    # reverse order matters as we change the index after the one we remove.
    for index_to_remove in reversed(dashes_to_remove):
        pruned_string = pruned_string[:index_to_remove] + pruned_string[index_to_remove + 1:]

    return pruned_string


class BuyEntry:
    def __init__(self, entry_id, name, description, manufacturer, price):
        self.manufacturer = manufacturer
        self.price = price
        self.description = description
        self.name = name
        self.entry_id = entry_id

    def __str__(self) -> str:
        return 'id:{}, name:{}, description:{}, manufacturer:{}, price:{}' \
            .format(self.entry_id, self.name, self.description, self.manufacturer, self.price)

    def transform_to_bag_of_words(self):
        # manual data analysis has shown that a lot of good keys are lost because they get splitted by a '-'.
        name_pruned = concat_dasherized_expressions(self.name)
        return bag_of_words(" ".join([name_pruned, self.description, self.manufacturer]))

    def transform_to_bag_of_words_name(self):
        # manual data analysis has shown that a lot of good keys are lost because they get splitted by a '-'.
        name_pruned = concat_dasherized_expressions(self.name)
        return bag_of_words(name_pruned)

    def transform_to_clean_string(self):
        return ' '.join(self.transform_to_bag_of_words())
