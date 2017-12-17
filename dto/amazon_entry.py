from preprocessing.preprocessing import bag_of_words


class AmazonEntry:
    def __init__(self, entry_id, title, description, manufacturer, price):
        self.price = price
        self.manufacturer = manufacturer
        self.description = description
        self.title = title
        self.entry_id = entry_id

    def __str__(self) -> str:
        return 'id:{}, title:{}'.format(self.entry_id, self.title)

    def transform_to_bag_of_words(self):
        return bag_of_words(" ".join([self.title, self.description, self.manufacturer]))

    def transform_to_clean_string(self):
        return ' '.join(self.transform_to_bag_of_words())
