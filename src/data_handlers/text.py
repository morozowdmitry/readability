import random
from typing import Optional, Iterable, Set, Dict, Any, List
from enum import Enum

import statistics
import numpy as np

from src.data_handlers.morphemes import MorphemeParsing
from src.data_handlers.frequency import Frequency
from src.data_handlers.syntax import SyntaxParsing
from src.label.labels import LabelType
from src.const import Language

from src.config import MIN_CLASS_SIZE
import random


class TokenType(Enum):
    WORD = "WORD"
    PUNCT = "PUNCT"


class Token(object):
    def __init__(
            self,
            wordform: str,
            token_type: Optional[TokenType] = None,
            lex: Optional[str] = None,
            pos: Optional[str] = None,  # FIXME pos are from fixed list of constants, should change
            morph: Optional[Iterable[str]] = None,  # FIXME grammars are from fixed list of constants, should change
            syntax: Optional[SyntaxParsing] = None,  # FIXME syntax are from fixed list of constants, should change
            morphemes: Optional[MorphemeParsing] = None,
            frequency: Optional[Frequency] = None
    ):
        self.wordform = wordform
        self.token_type = token_type
        self.lex = lex
        self.pos = pos
        self.morph = morph
        self.syntax = syntax
        self.morphemes = morphemes
        self.frequency = frequency if frequency is not None else Frequency()


class Sentence(object):
    def __init__(self, tokens: List[Token] = None):
        self.tokens = list() if tokens is None else tokens
        self.cached = {
            'words_number': -1
        }

    def words_number(self):
        if self.cached['words_number'] == -1:
            self.cached['words_number'] = len([_t for _t in self.tokens if _t.token_type == TokenType.WORD])
        return self.cached['words_number']


class Text(object):
    def __init__(
            self,
            text: str,
            true_label: str = None,
            language: Language = Language.RUS,
            sentences: List[Sentence] = None,
            labels: Optional[Set[LabelType]] = None,
            features: Optional[Dict[str, float]] = None,
    ):
        self.text = text
        self.true_label = true_label
        self.language = language
        self.sentences = list() if sentences is None else sentences
        self.labels = set() if labels is None else labels
        self.features = dict() if features is None else features
        self.embedding = None
        self.topic = None
        self.cached = {
            'words_number': -1,
            'words_mean_length': -1,
            'words_median_length': -1,
            'sentences_number': -1,
            'sentences_mean_length': -1,
            'sentences_median_length': -1,
        }

    def word_lengths(self):
        return [len(_t.wordform) for _s in self.sentences for _t in _s.tokens if _t.token_type == TokenType.WORD]

    def words_number(self):
        if self.cached['words_number'] == -1:
            words_number = len([
                _t
                for _s in self.sentences
                for _t in _s.tokens
                if _t.token_type == TokenType.WORD
            ])
            self.cached['words_number'] = words_number if words_number else 1
        return self.cached['words_number']

    def words_mean_length(self):
        if self.cached['words_mean_length'] == -1:
            self.cached['words_mean_length'] = statistics.mean(self.word_lengths())
        return self.cached['words_mean_length']

    def words_median_length(self):
        if self.cached['words_median_length'] == -1:
            self.cached['words_median_length'] = statistics.median(self.word_lengths())
        return self.cached['words_median_length']

    def words_max_length(self):
        return max(self.word_lengths())

    def words_25_75_length(self):
        return np.percentile(self.word_lengths(), [25, 75], method='midpoint')

    def words_sample(self):
        words = [
            _t
            for _s in self.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.WORD
        ]
        if not MIN_CLASS_SIZE:
            return words
        return random.sample(words, MIN_CLASS_SIZE)

    def sentence_lengths(self):
        return [_s.words_number() for _s in self.sentences]

    def sentences_number(self):
        if self.cached['sentences_number'] == -1:
            self.cached['sentences_number'] = len(self.sentences)
        return self.cached['sentences_number']

    def sentences_mean_length(self):
        if self.cached['sentences_mean_length'] == -1:
            self.cached['sentences_mean_length'] = statistics.mean(self.sentence_lengths())
        return self.cached['sentences_mean_length']

    def sentences_median_length(self):
        if self.cached['sentences_median_length'] == -1:
            self.cached['sentences_median_length'] = statistics.median(self.sentence_lengths())
        return self.cached['sentences_median_length']
