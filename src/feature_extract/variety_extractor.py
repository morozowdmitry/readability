from src.data_handlers.text import Text
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType


class NAVExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.MORPH, LabelType.LEMMA}
        self.extractor_label = 'NAV'

    def _count_features(self, text: Text) -> Text:
        all_nouns = list()
        all_adj = list()
        all_verbs = list()

        for sentence in text.sentences:
            for token in sentence.tokens:
                if token.pos in {'NOUN', 'n'}:
                    all_nouns.append(token.lex)
                elif token.pos in {'VERB', 'INFN', 'v'}:
                    all_verbs.append(token.lex)
                elif token.pos in {'ADJF', 'ADJS', 'a'}:
                    all_adj.append(token.lex)

        ttr_n = len(set(all_nouns)) / len(all_nouns) if len(all_nouns) > 0 else -1
        ttr_a = len(set(all_adj)) / len(all_adj) if len(all_adj) > 0 else -1
        ttr_v = len(set(all_verbs)) / len(all_verbs) if len(all_verbs) > 0 else -1

        nav_value = ttr_a * ttr_v / ttr_n if ttr_n > 0 else -1
        self._add_feature(text, 'nav', nav_value)

        return text


class LexicalVarietyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.LEMMA}
        self.extractor_label = 'LEXICAL_VARIETY'

    def _count_features(self, text: Text) -> Text:
        lemmas = list()
        for token in text.words_sample():
            lemmas.append(token.lex)
        self._add_feature(text, 'lexical_variety', len(set(lemmas)) / len(lemmas))
        return text
