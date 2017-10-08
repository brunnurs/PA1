from preprocessing.preprocessing import bag_of_words


class AbtEntry:
    def __init__(self, entry_id, name, description, price):
        self.price = price
        self.description = description
        self.name = name
        self.entry_id = entry_id

    def __str__(self) -> str:
        return 'id:{}, name:{}'.format(self.entry_id, self.name)

    def transform_to_bag_of_words(self):
        return bag_of_words(' '.join([self.name, self.description, self.price]))

    def transform_to_clean_string(self):
        return ' '.join(self.transform_to_bag_of_words())
