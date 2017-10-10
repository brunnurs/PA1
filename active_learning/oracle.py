class Oracle:
    def __init__(self, gold_standard):
        self.gold_standard = gold_standard
        self.interactions_with_oracle = 0

        print('initialize the oracle with a gold-standard of {} real matches'.format(len(gold_standard)))

    def is_match(self, abt_record_id, buy_record_id):
        self.interactions_with_oracle += 1
        return any(t for t in self.gold_standard if
                   t['abt_record_id'] == abt_record_id and t['buy_record_id'] == buy_record_id)

    def reset_interactions(self):
        self.interactions_with_oracle = 0