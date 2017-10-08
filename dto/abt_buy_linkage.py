class AbtBuyLinkage:
    def __init__(self, abt_entry, buy_entry):
        self.abt_entry = abt_entry
        self.buy_entry = buy_entry

    def __str__(self) -> str:
        return '[abt]:{} =========== matches entity =========== [buy]:{}'.format(self.abt_entry, self.buy_entry)
