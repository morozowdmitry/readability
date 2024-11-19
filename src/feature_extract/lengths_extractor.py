from typing import Dict, Any

from src.data_handlers.text import Text, LabelType
from src.feature_extract.base_extractor import BaseExtractor


class LengthsExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.LEMMA}
        self.extractor_label = 'LENGTHS'

    def _count_features(self, text: Text) -> Text:
        features = dict()
        features.update(self._sentences_length(text))
        features.update(self._words_length(text))
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text


    def _sentences_length(self, text: Text) -> Dict[str, Any[int, float]]:
        sent_lengths = text.sentence_lengths()
        return self._stats(sent_lengths, label='sentence_length')

    def _words_length(self, text: Text) -> Dict[str, Any[int, float]]:
        word_lengths = text.word_lengths()
        return self._stats(word_lengths, label='word_length')
