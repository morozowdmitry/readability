from typing import List
import pickle

from src.vectorize.vectorizer import BaseVectorizer
from src.data_handlers.text import Text, TokenType

import gensim
import gensim.corpora as corpora


class LDAVectorizer(BaseVectorizer):
    def fit_transform(self, texts: List[Text]):
        preprocessed = [self._preprocess_text(_t) for _t in texts]
        texts_bigrams = self._make_bigrams(preprocessed)

        id2word = corpora.Dictionary(texts_bigrams)
        corpus = [id2word.doc2bow(text) for text in texts_bigrams]

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=100,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               workers=3,
                                               per_word_topics=True)

        self.path.parent.mkdir(exist_ok=True)
        with self.path.open('wb') as f:
            pickle.dump(lda_model, f)
        with self.path.with_suffix('.dict.pt').open('wb') as f:
            pickle.dump(id2word, f)

        topics = [self._get_dist(lda_model.get_document_topics(_t, minimum_probability=0.0)) for _t in corpus]
        return topics

    def transform(self, texts: List[Text]):
        with self.path.open('rb') as f:
            lda_model = pickle.load(f)
        with self.path.with_suffix('.dict.pt').open('rb') as f:
            id2word = pickle.load(f)

        preprocessed = [self._preprocess_text(_t) for _t in texts]
        texts_bigrams = self._make_bigrams(preprocessed)
        corpus = [id2word.doc2bow(text) for text in texts_bigrams]

        topics = [self._get_dist(lda_model.get_document_topics(_t, minimum_probability=0.0)) for _t in corpus]
        return topics

    @staticmethod
    def _get_dist(dist):
        return [_d[1] for _d in dist]

    @staticmethod
    def _make_bigrams(texts: List[List[str]]):
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        return [bigram_mod[doc] for doc in texts]
