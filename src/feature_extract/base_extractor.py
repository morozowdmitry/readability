from typing import Set, Dict, Any

import statistics

from src.data_handlers.text import Text
from src.label.labels import LabelType

from src.config import MIN_CLASS_SIZE
import random


class LabelsMissingException(BaseException):
    pass


class BaseExtractor(object):
    def __init__(self):
        self.required_labels = set()
        self.extractor_label = 'BASE'

    def run(self, text: Text) -> Text:
        self._validate_labels(text_labels=text.labels)
        self._count_features(text=text)
        return text

    def _validate_labels(self, text_labels: Set[LabelType]):
        if any(x not in text_labels for x in self.required_labels):
            raise LabelsMissingException

    def _count_features(self, text: Text) -> Text:
        return text

    @staticmethod
    def _stats(accumulated, label, sampling=True) -> Dict[str, Any[int, float]]:
        if sampling and MIN_CLASS_SIZE:
            sampled = random.sample(accumulated, MIN_CLASS_SIZE)
        else:
            sampled = accumulated
        q1, q2, q3 = statistics.quantiles(sampled) if len(sampled) > 1 else [0, 0, 0]
        return {
            f'average_{label}': statistics.mean(sampled) if len(sampled) >= 1 else 0,
            f'25_perc_{label}': q1,
            f'median_{label}': q2,
            f'75_perc_{label}': q3,
            f'max_{label}': max(sampled) if len(sampled) >= 1 else 0
        }

    def _add_feature(self, text: Text, feature_name: str, feature_value: float):
        text.features[f'{self.extractor_label}#{feature_name}'] = feature_value
