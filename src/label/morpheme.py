import json

from src.label.base_labeler import BaseLabeler
from src.label.labels import LabelType
from src.data_handlers.text import TokenType
from src.data_handlers.morphemes import MorphemeParsing, Morpheme, MorphemeType
from src.config import MORPHEME_CONFIG_PATH, MORPHODICT_PATH
from src.label.NeuralMorphemeSegmentation.neural_morph_segm import load_cls


class MorphemeCNNLabeler(BaseLabeler):
    def __init__(self):
        super().__init__()
        self.labels = {LabelType.MORPHEME}
        self.model = load_cls(MORPHEME_CONFIG_PATH)
        self.morphodict = self._load_morphodict(morphodict_path=MORPHODICT_PATH)

    @staticmethod
    def _load_morphodict(morphodict_path):
        morphodict = json.load(open(morphodict_path, 'r'))
        for lex in morphodict.keys():
            morphemes = list()
            for _m in morphodict[lex].split('/'):
                morpheme_text, morpheme_label = _m.split(':')
                morpheme_label = MorphemeType(morpheme_label)
                morphemes.append(Morpheme(label=morpheme_label, text=morpheme_text))
            morphodict[lex] = MorphemeParsing(morphemes=morphemes)
        return morphodict

    def _label(self, text, labels):
        for sentence in text.sentences:
            for token in sentence.tokens:
                if token.token_type == TokenType.PUNCT:
                    continue
                self._parse_token(token)
        return text

    def _parse_token(self, token):
        if token.lex not in self.morphodict:
            morphemes = list()
            labels, _ = self.model._predict_probs([token.lex])[0]
            morpheme_labels, morpheme_types = self.model.labels_to_morphemes(
                token.lex, labels, return_probs=False, return_types=True
            )
            for morpheme_text, morpheme_label in zip(morpheme_labels, morpheme_types):
                morphemes.append(Morpheme(label=MorphemeType(morpheme_label), text=morpheme_text))
            self.morphodict[token.lex] = MorphemeParsing(morphemes=morphemes)
        token.morphemes = self.morphodict[token.lex]
