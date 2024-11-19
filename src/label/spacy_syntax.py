import spacy
from spacy.tokens import Doc

from src.label.base_labeler import BaseLabeler
from src.label.labels import LabelType
from src.data_handlers.text import Text
from src.data_handlers.syntax import SyntaxParsing


class SpacySyntaxLabeler(BaseLabeler):
    def __init__(self):
        super().__init__()
        self.labels = {LabelType.SYNTAX}
        self.spacy_pipeline = spacy.load('ru_core_news_lg')

    def _label(self, text, labels) -> Text:
        for sentence in text.sentences:
            doc = self.spacy_pipeline(Doc(self.spacy_pipeline.vocab,
                                          words=[x.wordform for x in sentence.tokens],
                                          sent_starts=[True] + [False] * (len(sentence.tokens) - 1)))
            for sent in doc.sents:
                for token, spacy_token in zip(sentence.tokens, sent):
                    token.syntax = SyntaxParsing(dep=spacy_token.dep_,
                                                 idx=spacy_token.i,
                                                 head_idx=spacy_token.head.i,
                                                 children_idx=[c.i for c in spacy_token.children],
                                                 is_root=spacy_token.i == sent.root.i)
        return text
