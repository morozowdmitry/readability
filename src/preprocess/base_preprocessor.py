from src.data_handlers.text import Text


class BasePreprocessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def run(self, text: str) -> Text:
        tokenized = self.tokenizer.tokenize(text)
        return tokenized
