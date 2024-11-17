from collections import Counter

from src.data_handlers.text import Text, TokenType
from src.feature_extract.base_extractor import BaseExtractor
from src.label.labels import LabelType


class MorphologyExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.required_labels = {LabelType.MORPH}
        self.extractor_label = 'MORPHOLOGY'

    def _count_features(self, text: Text) -> Text:
        features = dict()
        features.update(self._pos_ratio(text))
        features.update(self._case_ratio(text))
        features.update(self._anim_ratio(text))
        features.update(self._verb_form_ratio(text))
        features.update(self._verb_tense_ratio(text))
        features.update(self._verb_tran_ratio(text))
        for k, v in features.items():
            self._add_feature(text, k, v)
        return text

    @staticmethod
    def _pos_ratio(text: Text) -> dict:
        pos_number = Counter([_t.pos for _t in text.words_sample()])
        return {
            'NOUN_ratio': pos_number['NOUN'] / text.words_number(),
            'ADJ_ratio': (pos_number['ADJF'] + pos_number['ADJS']) / text.words_number(),
            'VERB_ratio': (pos_number['VERB'] + pos_number['INFN'] +
                           pos_number['PRTF'] + pos_number['PRTS'] + pos_number['GRND']) / text.words_number(),
            'NUMR_ratio': pos_number['NUMR'] / text.words_number(),
            'ADVB_ratio': pos_number['ADVB'] / text.words_number(),
            'NPRO_ratio': pos_number['NPRO'] / text.words_number(),
            'PRED_ratio': pos_number['PRED'] / text.words_number(),
            'PREP_ratio': pos_number['PREP'] / text.words_number(),
            'CONJ_ratio': pos_number['CONJ'] / text.words_number(),
            'PRCL_ratio': pos_number['PRCL'] / text.words_number(),
            'INTJ_ratio': pos_number['INTJ'] / text.words_number(),
        }

    @staticmethod
    def _case_ratio(text: Text) -> dict:
        cases = ['nomn', 'gent', 'datv', 'accs', 'ablt', 'loct', 'voct', 'gen2', 'acc2', 'loc2']
        case_number = Counter([
            _t.morph.case
            for _t in text.words_sample()
            if _t.morph.case is not None
        ])
        total_case_number = sum(case_number.values()) if sum(case_number.values()) > 0 else 1
        return {
            k: case_number[k] / total_case_number for k in cases
        }

    @staticmethod
    def _anim_ratio(text: Text) -> dict:
        anim_number = Counter([
            _t.morph.animacy
            for _t in text.words_sample()
            if _t.morph.animacy is not None
        ])
        total_anim_number = sum(anim_number.values()) if sum(anim_number.values()) > 0 else 1
        return {
            'anim_ratio': anim_number['anim'] / total_anim_number
        }

    @staticmethod
    def _verb_form_ratio(text: Text) -> dict:
        verb_form_number = Counter([
            _t.pos
            for _t in text.words_sample()
            if _t.pos in ['VERB', 'INFN', 'PRTF', 'PRTS', 'GRND']
        ])
        total_verb_number = sum(verb_form_number.values()) if sum(verb_form_number.values()) > 0 else 1
        return {
            k: verb_form_number[k] / total_verb_number for k in ['VERB', 'INFN', 'PRTF', 'PRTS', 'GRND']
        }

    @staticmethod
    def _verb_tense_ratio(text: Text) -> dict:
        tenses = ['pres', 'past', 'futr']
        tense_number = Counter([
            _t.morph.tense
            for _t in text.words_sample()
            if _t.pos in ['VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'] and _t.morph.tense is not None
        ])
        total_tense_number = sum(tense_number.values()) if sum(tense_number.values()) else 1
        return {
            k: tense_number[k] / total_tense_number for k in tenses
        }

    @staticmethod
    def _verb_tran_ratio(text: Text) -> dict:
        tran_number = Counter([
            _t.morph.transitivity
            for _t in text.words_sample()
            if _t.morph.transitivity is not None
        ])
        total_tran_number = sum(tran_number.values()) if sum(tran_number.values()) else 1
        return {
            'tran_ratio': tran_number['tran'] / total_tran_number
        }

