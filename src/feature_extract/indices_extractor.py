import statistics
from math import sqrt

from src.data_handlers.text import Text, Token, TokenType
from src.feature_extract.base_extractor import BaseExtractor
from src.const import Language
from src.label.labels import LabelType


class SyllablesExtractor(BaseExtractor):
    @staticmethod
    def _count_syllables(token: Token, language: Language) -> int:
        wordform = token.wordform.lower()
        vowels_en = "aeiouy"
        vowels_ru = 'аяуюэеоёиы'

        # if any([x in wordform for x in vowels_en]):
        if language == Language.ENG:
            count = 0
            if wordform[0] in vowels_en:
                count += 1
            for index in range(1, len(wordform)):
                if wordform[index] in vowels_en and wordform[index - 1] not in vowels_en:
                    count += 1
            if wordform.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
            return count
        elif language == Language.RUS:
            return len([x for x in wordform if x in vowels_ru])
        return -1


class IndicesExtractor(SyllablesExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.MORPH}
        self.extractor_label = 'INDICES'
        self.language_constants = {
            'flesch_kincaid': {
                Language.ENG: (0.38, 11.8, 15.59),  # from original paper
                Language.RUS: (0.5, 8.4, 15.59),  # from Oborneva (2005)
            },
            'coleman_liau': {
                Language.ENG: (0.0588, 0.296, 15.8),  # from original paper
                Language.RUS: (0.055, 0.35, 20.33),  # from plainrussian
            },
            'ari': {
                Language.ENG: (4.71, 0.5, 21.43),  # from original paper
                Language.RUS: (6.26, 0.2805, 31.04),  # from plainrussian
            },
            'smog': {
                Language.ENG: (1.043, 30., 3.1291),  # from original paper
                Language.RUS: (1.1, 64.6, 0.05),  # from plainrussian
            },
            'dave_chall': {
                Language.ENG: (0.1579, 0.0496),  # from original paper
                Language.RUS: (0.552, 0.273),  # from plainrussian
            }
        }

    def _count_features(self, text: Text) -> Text:
        self._flesch_kincaid(text=text)
        self._coleman_liau(text=text)
        self._ari(text=text)
        self._smog(text=text)
        self._dave_chall(text=text)
        return text

    def _flesch_kincaid(self, text: Text) -> Text:
        avg_sent_length = text.sentences_mean_length()
        word2syllables = [
            self._count_syllables(_t, text.language)
            for _s in text.sentences
            for _t in _s.tokens
            if _t.pos is not None
        ]
        avg_syllables = statistics.mean(word2syllables) if word2syllables else 0
        consts = self.language_constants['flesch_kincaid'][text.language]
        index_value = consts[0] * avg_sent_length + consts[1] * avg_syllables - consts[2]
        self._add_feature(text, 'flesch_kincaid', index_value)
        return text

    def _coleman_liau(self, text: Text) -> Text:
        scalar = len(text.word_lengths()) / 100
        letters_per_100 = sum(text.word_lengths()) / scalar
        sents_per_100 = text.sentences_number() / scalar
        consts = self.language_constants['coleman_liau'][text.language]
        index_value = consts[0] * letters_per_100 - consts[1] * sents_per_100 - consts[2]
        self._add_feature(text, 'coleman_liau', index_value)
        return text

    def _ari(self, text: Text) -> Text:
        avg_word_length = text.words_mean_length()
        avg_sent_length = text.sentences_mean_length()
        consts = self.language_constants['ari'][text.language]
        index_value = consts[0] * avg_word_length + consts[1] * avg_sent_length - consts[2]
        self._add_feature(text, 'ari', index_value)
        return text

    def _smog(self, text: Text) -> Text:
        num_long = len([
            x
            for sentence in text.sentences
            for x in sentence.tokens
            if x.pos is not None and self._count_syllables(x, text.language) > 4
        ])
        num_sents = len(text.sentences)
        consts = self.language_constants['smog'][text.language]
        index_value = consts[0] * sqrt((consts[1] / num_sents) * num_long) + consts[2]
        self._add_feature(text, 'smog', index_value)
        return text

    def _dave_chall(self, text: Text) -> Text:
        num_long = len([
            x
            for sentence in text.sentences
            for x in sentence.tokens
            if x.pos is not None and self._count_syllables(x, text.language) > 4
        ])
        num_sents = text.sentences_number()
        num_words = text.words_number()
        consts = self.language_constants['dave_chall'][text.language]
        index_value = consts[0] * (100.0 * num_long / num_words) + consts[1] * (float(num_words) / num_sents)
        self._add_feature(text, 'dave_chall', index_value)
        return text


class LongWordsExtractor(SyllablesExtractor):
    def __init__(self):
        super().__init__()
        self.extractor_label = 'LONG_WORDS'

    def _count_features(self, text: Text) -> Text:
        tokens = [
            token
            for sentence in text.sentences
            for token in sentence.tokens
            if token.token_type == TokenType.WORD
        ]
        long_tokens = [x for x in tokens if self._count_syllables(x, text.language) > 4]
        self._add_feature(text, 'long_words_ratio', float(len(long_tokens)) / len(tokens))
        return text
