class Oracle:
    def __init__(self, gold_standard):
        self.gold_standard = gold_standard
        self.interactions_with_oracle = 0

        print('initialize the oracle with a gold-standard of {} real matches'.format(len(gold_standard)))

    def is_match(self, record_a_id, record_b_id):
        self.interactions_with_oracle += 1
        return any(t for t in self.gold_standard if
                   t['record_a_id'] == record_a_id and t['record_b_id'] == record_b_id)

    def reset_interactions(self):
        self.interactions_with_oracle = 0