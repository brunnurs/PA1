from preprocessing.preprocessing import bag_of_words


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
        return bag_of_words(" ".join([self.name, self.description, self.manufacturer]))

    def transform_to_clean_string(self):
        return ' '.join(self.transform_to_bag_of_words())
