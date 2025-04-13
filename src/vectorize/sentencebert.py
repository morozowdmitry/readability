from typing import List
from pathlib import Path

from src.data_handlers.text import Text, TokenType
from src.vectorize.vectorizer import BaseVectorizer

from sentence_transformers import SentenceTransformer


class SentenceBertVectorizer(BaseVectorizer):
    def __init__(self, model_path: Path):
        super(SentenceBertVectorizer, self).__init__(model_path=model_path)
        self.embedder = SentenceTransformer('distiluse-base-multilingual-cased')


    def fit_transform(self, texts: List[Text]):
        embeddings = self.embedder.encode([_t.text for _t in texts])
        return embeddings, [f"SBEmb_{str(idx).zfill(3)}" for idx in range(embeddings.shape[1])]

    def transform(self, texts: List[Text]):
        embeddings = self.embedder.encode([_t.text for _t in texts])
        return embeddings, [f"SBEmb_{str(idx).zfill(3)}" for idx in range(embeddings.shape[1])]

    def transform_keywords(self, texts: List[str]):
        embeddings = self.embedder.encode([_t.text for _t in texts])
        return embeddings, [f"SBEmb_{str(idx).zfill(3)}" for idx in range(embeddings.shape[1])]
