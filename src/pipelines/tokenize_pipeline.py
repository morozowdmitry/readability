from typing import List

import json
import pandas as pd

from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.exceptions import NoDataError
from src.convertor.convert2df import texts2df
from src.data_handlers.text import Text


class TokenizePipeline(BasePipeline):
    def run(self, raw_texts: List[str] = None):
        if raw_texts is None:
            raise NoDataError
        tokenized, prelabeled_df = self._prepare_data(raw_texts=raw_texts)
        tokenized_df = self._serialize(tokenized)
        tokenized_df.to_csv(self.config.labeled_path, index=False)

    @staticmethod
    def _serialize(tokenized: List[Text]) -> pd.DataFrame:
        serialized = list()
        for text in tokenized:
            text_serialized = list()
            for sent in text.sentences:
                sentence_serialized = list()
                for token in sent.tokens:
                    sentence_serialized.append(f'{token.wordform}:{token.token_type.value}')
                text_serialized.append(sentence_serialized)
            serialized.append(json.dumps(text_serialized, ensure_ascii=False))
        return pd.DataFrame(serialized, columns=["tokenized_text"])

