from typing import Set

import csv

from src.label.base_labeler import BaseLabeler
from src.label.labels import LabelType
from src.data_handlers.text import TokenType, Text
from src.data_handlers.frequency import Frequency, FrequencyDict
from src.const import Language
from src.config import FREQUENCY_RUS_PATH, FREQUENCY_ENG_PATH


class FrequencyLabeler(BaseLabeler):
    def __init__(self):
        super().__init__()
        self.labels = {LabelType.FREQUENCY}
        self.dicts = {
            Language.RUS: self._load_rnc(),
            Language.ENG: self._load_bnc(),
        }

    @staticmethod
    def _load_rnc():
        rnc_dict_reader = csv.DictReader(open(FREQUENCY_RUS_PATH, 'r'), delimiter='\t')
        freq_dict = FrequencyDict()

        rnc2pos = {
            's': 'NOUN',
            'a': 'ADJ',
            'v': 'VERB',
            'adv': 'ADVB'
        }

        for row in rnc_dict_reader:
            if row['Lemma'] in freq_dict.frequencies:
                freq_dict.frequencies[row['Lemma']].frequency += float(row['Freq(ipm)'])
            else:
                freq_dict.frequencies[row['Lemma']] = Frequency(frequency=float(row['Freq(ipm)']))
            if row['PoS'] in rnc2pos:
                pos = rnc2pos[row["PoS"]]
                freq_dict.frequencies[f'{row["Lemma"]}_{pos}'] = Frequency(frequency=float(row['Freq(ipm)']))
        freq_dict.evaluate_categories()
        return freq_dict

    # TODO add code for English texts
    def _load_bnc(self):
        return

    def _label(self, text: Text, labels: Set[LabelType]):
        for _s in text.sentences:
            for _t in _s.tokens:
                _t.frequency = self.dicts[text.language].frequency(_t)
