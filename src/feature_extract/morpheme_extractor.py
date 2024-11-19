from src.data_handlers.text import Text, TokenType
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType
from src.label.morpheme import MorphemeType


class LazyMorphemeExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.LEMMA}
        self.extractor_label = 'LAZY_MORPHEME'

    def _count_features(self, text: Text) -> Text:
        features = dict()

        for ending in ['ция', 'ние', 'вие', 'тие', 'ист', 'изм', 'ура', 'ище',
                       'ство', 'ость', 'овка', 'атор', 'итор', 'тель', 'льный', 'овать']:
            features[ending] = len([
                _t
                for _s in text.sentences
                for _t in _s.tokens
                if _t.token_type == TokenType.WORD and _t.lex.endswith(ending)
            ]) / text.words_number()
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text


class MorphemeVarietyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.MORPHEME, LabelType.LEMMA}
        self.extractor_label = 'MORPHEME'

    def _count_features(self, text: Text) -> Text:
        features = dict()
        features.update(self._morphemes_num(text))
        features.update(self._roots_num(text))
        features.update(self._suffix_length(text))
        features.update(self._morphemes_variety(text))
        features.update(self._has_link_ratio(text))
        features.update(self._difficult_suffix_ratio(text))
        # features.update(self._root_popularity(text))
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text

    def _morphemes_num(self, text: Text):
        morphemes_num = [
            len(_t.morphemes.morphemes)
            for _s in text.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.WORD
        ]
        return self._stats(morphemes_num, label="morphemes")

    def _roots_num(self, text: Text):
        roots_num = [
            len([_m for _m in _t.morphemes.morphemes if _m.label == MorphemeType.ROOT])
            for _s in text.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.WORD
        ]
        return self._stats(roots_num, label="roots")

    @staticmethod
    def _morphemes_variety(text: Text):
        unique_morphemes = set()
        unique_roots = set()
        total_morphemes = 0
        total_roots = 0
        samples = text.words_sample()
        for _t in samples:
            if _t.token_type != TokenType.WORD:
                continue
            for _m in _t.morphemes.morphemes:
                total_morphemes += 1
                unique_morphemes.add(f'{_m.text}_{_m.label}')
                if _m.label == MorphemeType.ROOT:
                    total_roots += 1
                    unique_roots.add(_m.text)
        return {
            'unique_morphemes_ratio': len(unique_morphemes) / total_morphemes,
            'unique_roots_ratio': len(unique_roots) / total_roots
        }

    def _suffix_length(self, text: Text):
        suff_len = [
            len(''.join([_m.text for _m in _t.morphemes.morphemes if _m.label == MorphemeType.SUFF]))
            for _s in text.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.WORD
        ]
        return self._stats(suff_len, label="suffix_lengths")

    @staticmethod
    def _difficult_suffix_ratio(text: Text):
        difficult_suffixes = [
            'ни', 'ени', 'ц', 'ость', 'ств', 'ящ', 'ющ', 'еск',
            'ист', 'изм', 'лищ', 'атор', 'тор', 'тель', 'ирова', 'иров'
        ]
        sample = text.words_sample()
        has_difficult_suffix = [
            _t
            # for _s in text.sentences
            # for _t in _s.tokens
            # if _t.token_type == TokenType.WORD and
            for _t in sample if
               any(_m.label == MorphemeType.SUFF and _m.text in difficult_suffixes for _m in _t.morphemes.morphemes)
        ]
        # return {'difficult_suffixes_ratio': len(has_difficult_suffix) / text.words_number()}
        return {'difficult_suffixes_ratio': len(has_difficult_suffix) / len(sample)}

    @staticmethod
    def _has_link_ratio(text: Text):
        sample = text.words_sample()
        has_link = [
            _t
            # for _s in text.sentences
            # for _t in _s.tokens
            # if _t.token_type == TokenType.WORD and
            for _t in sample if
               any(_m.label == MorphemeType.LINK for _m in _t.morphemes.morphemes)
        ]
        # return {'link_ratio': len(has_link) / text.words_number()}
        return {'link_ratio': len(has_link) / len(sample)}

    @staticmethod
    def _root_popularity(text: Text):
        return {}

