from typing import Optional, Dict

import statistics

from src.data_handlers.text import Text, PoS, COMPLEX_POS
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType


class FrequencyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.FREQUENCY, LabelType.LEMMA, LabelType.MORPH}
        self.extractor_label = 'FREQUENCY'

    def _count_features(self, text: Text) -> Text:
        features = dict()
        features.update(self._avg_frequency(text))
        features.update(self._ratio_frequency(text))
        for pos in [PoS.NOUN, PoS.VERB, PoS.ADVB, PoS.ADJ]:
            features.update(self._ratio_frequency(text, pos=pos))
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text

    @staticmethod
    def _avg_frequency(text: Text) -> Dict[str, int | float]:
        freq_list = [
            _t.frequency.frequency
            for _t in text.words_sample()
        ]
        q1, q2, q3 = statistics.quantiles(freq_list)
        return {
            'mean_frequency': statistics.mean(freq_list),
            'min_frequency': min(freq_list),
            '25_perc_frequency': q1,
            'median_frequency': q2,
            '75_perc_frequency': q3,
        }

    @staticmethod
    def _ratio_frequency(text: Text, pos: Optional[PoS] = None) -> Dict[str, int | float]:
        ratios = dict()

        if pos is not None:
            for i in range(10):
                freq_enough = [
                    _t for _t in text.words_sample()
                    if _t.pos in COMPLEX_POS.get(pos, set()) and _t.frequency.pos_category == i
                ]
                all_pos = [
                    _t
                    for _t in text.words_sample()
                    if _t.pos in COMPLEX_POS.get(pos, set())
                ]
                ratios[f'ratio_{i}_{pos.value}'] = len(freq_enough) / len(all_pos) if len(all_pos) > 0 else 0
        else:
            for i in range(10):
                sample = text.words_sample()
                freq_enough = [_t for _t in sample if _t.frequency.category == i]
                ratios[f'ratio_{i}'] = len(freq_enough) / len(sample)

        return ratios
