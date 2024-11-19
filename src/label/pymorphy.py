import pymorphy2
from typing import Set

from src.label.base_labeler import BaseLabeler
from src.label.labels import LabelType
from src.data_handlers.text import Text, TokenType


class PymorphyLemmatizer(BaseLabeler):
    def __init__(self):
        super().__init__()
        self.labels = {LabelType.LEMMA, LabelType.MORPH}
        self.morph_analyzer = pymorphy2.MorphAnalyzer()
        self.lemmas_dictionary = dict()

    def _label(self, text, labels) -> Text:
        for sentence in text.sentences:
            for token in sentence.tokens:
                self._parse_token(token, labels)
        return text

    def _parse_token(self, token, labels: Set[LabelType]):
        if token.wordform not in self.lemmas_dictionary:
            self.lemmas_dictionary[token.wordform] = dict()
            p = self.morph_analyzer.parse(token.wordform)[0]
            if 'PNCT' in p.tag:
                self.lemmas_dictionary[token.wordform]['token_type'] = TokenType.PUNCT
            else:
                self.lemmas_dictionary[token.wordform]['token_type'] = TokenType.WORD
                self.lemmas_dictionary[token.wordform]['lex'] = p.normal_form
                self.lemmas_dictionary[token.wordform]['pos'] = p.tag.POS
                self.lemmas_dictionary[token.wordform]['morph'] = p.tag
        if self.lemmas_dictionary[token.wordform]['token_type'] == TokenType.PUNCT:
            token.token_type = TokenType.PUNCT
        else:
            token.token_type = TokenType.WORD
            if LabelType.LEMMA in labels:
                token.lex = self.lemmas_dictionary[token.wordform]['lex']
            if LabelType.MORPH in labels:
                token.pos = self.lemmas_dictionary[token.wordform]['pos']
                token.morph = self.lemmas_dictionary[token.wordform]['morph']


