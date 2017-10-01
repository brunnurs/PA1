class BuyEntry:
    def __init__(self, entry_id, name, description, manufacturer, price):
        self.manufacturer = manufacturer
        self.price = price
        self.description = description
        self.name = name
        self.entry_id = entry_id

    def __str__(self) -> str:
        return 'id:{}, name:{}'.format(self.entry_id, self.name)
