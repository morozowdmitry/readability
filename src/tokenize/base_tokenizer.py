from src.data_handlers.text import Text


class BaseTokenizer(object):
    @staticmethod
    def tokenize(text: str) -> Text:
        return Text(text=text)
