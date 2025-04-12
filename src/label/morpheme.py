import json
from simpletransformers.ner import NERModel
import pandas as pd

from src.label.base_labeler import BaseLabeler
from src.label.labels import LabelType
from src.data_handlers.text import TokenType
from src.data_handlers.morphemes import MorphemeParsing, Morpheme, MorphemeType
from src.config import MORPHEME_CONFIG_PATH, MORPHODICT_PATH


class MorphemeBERTLabeler(BaseLabeler):
    def __init__(self):
        super().__init__()
        self.labels = {LabelType.MORPHEME}
        self.model = NERModel(
            'roberta',
            MORPHEME_CONFIG_PATH,
        )
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
            eval_data = [(0, token.lex, '0')] + [(0, letter, '0') for letter in token.lex]
            eval_data = pd.DataFrame(
                eval_data, columns=["sentence_id", "words", "labels"]
            )
            result, model_outputs, preds_list = self.model.eval_model(eval_data, silent=True)
            morphemes = self._convert2parsing(token.lex, preds_list[0][1:])
            self.morphodict[token.lex] = MorphemeParsing(morphemes=morphemes)
        token.morphemes = self.morphodict[token.lex]

    @staticmethod
    def _convert2parsing(lemma, bmes):
        parsing = list()
        current_mtype = ''
        current_mtext = ''
        for letter, label in zip(lemma, bmes):
            pos = label[0]
            mtype = label[2:]
            if pos == 'S':
                if current_mtext:
                    parsing.append({"morpheme": current_mtext, "type": current_mtype})
                parsing.append({"morpheme": letter, "type": mtype})
                current_mtype = ''
                current_mtext = ''
            elif pos == 'B':
                if current_mtext:
                    parsing.append({"morpheme": current_mtext, "type": current_mtype})
                current_mtext = letter
                current_mtype = mtype
            else:
                current_mtext += letter
        if current_mtext:
            parsing.append({"morpheme": current_mtext, "type": current_mtype})
        return [Morpheme(label=MorphemeType(x["type"]), text=x["morpheme"]) for x in parsing]
