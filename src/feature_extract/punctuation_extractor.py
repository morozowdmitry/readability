from src.data_handlers.text import Text, TokenType
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType


class PunctuationExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.extractor_label = 'PUNCTUATION'

    def _count_features(self, text: Text) -> Text:
        self._add_feature(text, 'punct_token_ratio', self._punct_token_ratio(text))
        self._add_feature(text, 'semicolon_punct_ratio', self._semicolon_punct_ratio(text))
        return text

    @staticmethod
    def _punct_token_ratio(text):
        return len([
                _t
                for _s in text.sentences
                for _t in _s.tokens
                if _t.token_type == TokenType.PUNCT
            ]) / text.words_number()

    @staticmethod
    def _semicolon_punct_ratio(text):
        puncts = [
            _t
            for _s in text.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.PUNCT
        ]
        return len([x for x in puncts if x.wordform == ';']) / len(puncts) if len(puncts) > 0 else 0
