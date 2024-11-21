from typing import List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.pipelines.config import PipelineConfig
from src.data_handlers.text import Text
from src.label.labels import LabelType

from src.tokenize.base_tokenizer import BaseTokenizer
from src.preprocess.base_preprocessor import BasePreprocessor


class BasePipeline(object):
    def __init__(self, config: PipelineConfig):
        self.config = config

    def run(self, raw_texts: Optional[List[str]] = None):
        return

    def _prepare_data(self,
                      raw_texts: List[str],
                      prelabeled_df: Optional[pd.DataFrame] = None):
        processed_texts = list()

        filtered_extractors = list()
        all_extractor_labels = set()
        for extractor in self.config.extractors:
            extractor_label = extractor.extractor_label
            all_extractor_labels.add(extractor_label)
            if prelabeled_df is not None and any(c.startswith(extractor_label) for c in prelabeled_df.columns):
                continue
            filtered_extractors.append(extractor)
        if prelabeled_df is not None:
            prelabeled_df = prelabeled_df[
                [_c for _c in prelabeled_df.columns if any(_c.startswith(_l) for _l in all_extractor_labels)]
            ]

        required_labels = set()
        for feature_extractor in filtered_extractors:
            required_labels |= set(feature_extractor.required_labels)
        if self.config.topic_modeler and self.config.topics_path is None:
            required_labels.add(LabelType.LEMMA)
        if not self.config.extractors and self.config.preprocessor:
            required_labels.add(LabelType.LEMMA)

        if required_labels:
            for raw_text in tqdm(raw_texts):
                processed_text = self.config.preprocessor.run(raw_text)

                for labeler, labeler_labels in self.config.labelers:
                    if labeler_labels.intersection(required_labels):
                        labeler.run(processed_text, labeler_labels.intersection(required_labels))
                processed_texts.append(processed_text)

            print('texts labeled')
        else:
            preprocessor = BasePreprocessor(BaseTokenizer())
            for raw_text in raw_texts:
                processed_text = preprocessor.run(raw_text)
                processed_texts.append(processed_text)

        if filtered_extractors and processed_texts:
            for processed_text in tqdm(processed_texts):
                for feature_extractor in filtered_extractors:
                    feature_extractor.run(processed_text)
            print(f'features extracted (total {len(processed_texts[0].features)} features)')

        if self.config.vectorizer is not None and processed_texts:
            processed_texts = self._run_vectorizer(texts=processed_texts)

            print('texts vectorized')

        # TODO add KEYWORDS processing here

        if self.config.topic_modeler is not None and processed_texts:
            processed_texts = self._run_topic_modeler(texts=processed_texts)

            print('topics modeled')

        return processed_texts, prelabeled_df

    def _run_vectorizer(self, texts: List[Text]) -> List[Text]:
        raise NotImplementedError

    def _run_topic_modeler(self, texts: List[Text]) -> List[Text]:
        raise NotImplementedError

