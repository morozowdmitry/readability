from typing import Optional

import stanza

from src.data_handlers.text import Text, Sentence, Token
from src.tokenize.base_tokenizer import BaseTokenizer
from src.config import STANZA_PATH


class StanzaTokenizer(BaseTokenizer):
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(STANZA_PATH)
        self.ppln = stanza.Pipeline(
            lang='ru',
            processors='tokenize',
            tokenize_model_path=model_path,
            use_gpu=True,
            # download_method=None
        )

    def tokenize(self, text: str) -> Text:
        doc = self.ppln(text)
        sentences = list()
        for _s in doc.sentences:
            tokens = [Token(wordform=_w.text) for _w in _s.words]
            sentences.append(Sentence(tokens=tokens))
        return Text(text=text, sentences=sentences)
