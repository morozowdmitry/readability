from typing import List
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
import pickle

from src.data_handlers.text import Text, TokenType


import nltk
nltk.download('stopwords')  # NLTK Stop words from nltk.corpus
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')


class BaseVectorizer:
    def __init__(self, model_path: Path):
        self.path = model_path

    def fit_transform(self, texts: List[Text]):
        raise NotImplementedError

    def transform(self, texts: List[Text]):
        raise NotImplementedError

    def transform_keywords(self, texts: List[str]):
        raise NotImplementedError

    @staticmethod
    def _preprocess_text(text: Text):
        return [
            _t.lex
            for _s in text.sentences
            for _t in _s.tokens
            if _t.token_type == TokenType.WORD and _t.lex not in stop_words
        ]


class BoWVectorizer(BaseVectorizer):
    def fit_transform(self, texts: List[Text]):
        vectorizer = CountVectorizer(max_features=10000)
        embeddings = vectorizer.fit_transform([_t.text for _t in texts])
        self.path.parent.mkdir(exist_ok=True)
        with self.path.open('wb') as f:
            pickle.dump(vectorizer, f)
        return embeddings.toarray(), vectorizer.get_feature_names_out()

    def transform(self, texts: List[Text]):
        with self.path.open('rb') as f:
            vectorizer = pickle.load(f)
        embeddings = vectorizer.transform([_t.text for _t in texts])
        return embeddings.toarray(), vectorizer.get_feature_names_out()

    def transform_keywords(self, texts: List[str]):
        with self.path.open('rb') as f:
            vectorizer = pickle.load(f)
        embeddings = vectorizer.transform(texts)
        return embeddings.toarray(), vectorizer.get_feature_names_out()
