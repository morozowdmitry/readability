from src.data_handlers.text import Text, Sentence, Token
from src.tokenize.base_tokenizer import BaseTokenizer


class SplitTokenizer(BaseTokenizer):
    @staticmethod
    def tokenize(text: str) -> Text:
        sentences = list()
        # for _s in text.split('.'):
        tokens = [Token(wordform=x) for x in text.split()]
        sentences.append(Sentence(tokens=tokens))
        return Text(text=text, sentences=sentences)

