from collections import defaultdict


class Frequency(object):
    def __init__(self, frequency: float = 0.0, category=100, pos_category=100):
        self.frequency = frequency
        self.category = category
        self.pos_category = pos_category


class FrequencyDict(object):
    def __init__(self):
        self.frequencies = defaultdict(Frequency)

    def freq_value(self, token) -> float:
        return self.frequencies.get(token.lex, Frequency()).frequency

    def category(self, token) -> int:
        return self.frequencies.get(token.lex, Frequency()).category

    def frequency(self, token) -> Frequency:
        return self.frequencies.get(token.lex, Frequency())

    def evaluate_categories(self):
        all_lemmas = sorted([(k, v.frequency) for k, v in self.frequencies.items() if '_' not in k],
                            key=lambda x: x[1],
                            reverse=True)

        borders = list()
        for i in range(1, 11):
            borders.append(all_lemmas[i * 100][1])

        for k, v in self.frequencies.items():
            if '_' in k:
                continue
            v.category = len([_b for _b in borders if _b > v.frequency])

        self._evaluate_category_pos('NOUN')
        self._evaluate_category_pos('VERB')
        self._evaluate_category_pos('ADVB')
        self._evaluate_category_pos('ADJ')

    def _evaluate_category_pos(self, pos):
        pos_lemmas = sorted([(k, v.frequency) for k, v in self.frequencies.items() if k.endswith(pos)],
                            key=lambda x: x[1],
                            reverse=True)

        borders = list()
        for i in range(1, 11):
            borders.append(pos_lemmas[i * 100][1])

        for k, v in self.frequencies.items():
            if '_' in k:
                continue
            if f'{k}_{pos}' in self.frequencies:
                v.pos_category = len([_b for _b in borders if _b > v.frequency])

