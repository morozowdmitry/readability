from typing import List, Optional
import pandas as pd

from src.pipelines.base_pipeline import BasePipeline
from src.data_handlers.text import Text
from src.convertor.convert2df import labels2series
from src.pipelines.exceptions import NoDataError, LabelsRequiredError
from src.convertor.convert2df import texts2df


class TrainPipeline(BasePipeline):
    def run(self,
            raw_texts: List[str] = None,
            processed_texts: Optional[List[Text]] = None,
            true_labels: Optional[List[str]] = None):
        if raw_texts is None and processed_texts is None:
            raise NoDataError

        if self.config.labeled_path is not None:
            prelabeled_df = pd.read_csv(self.config.labeled_path)
        else:
            prelabeled_df = None

        if processed_texts is None:
            processed_texts, prelabeled_df = self._prepare_data(raw_texts=raw_texts, prelabeled_df=prelabeled_df)
        else:
            _, prelabeled_df = self._prepare_data(raw_texts=[], prelabeled_df=prelabeled_df)
        X = texts2df(processed_texts, prelabeled_df)

        if true_labels is None:
            raise LabelsRequiredError
        y = labels2series(true_labels)

        print('data ready, start training')
        self.config.predictor.train(X, y, model_path=self.config.model_path, scaler=self.config.scaler)
        print('training finished')

        return processed_texts

    def _run_vectorizer(self, texts: List[Text]) -> List[Text]:
        embeddings, feature_names = self.config.vectorizer.fit_transform(texts=texts)
        if self.config.vectorizer_train_only:
            return texts

        for _t, _e in zip(texts, embeddings):
            _t.embedding = {
                feature_name: feature_value
                for feature_name, feature_value in zip(feature_names, _e)
            }
        return texts

    def _run_topic_modeler(self, texts: List[Text]) -> List[Text]:
        if self.config.topics_path is not None:
            topics_df = pd.read_csv(self.config.topics_path)
            for text, (index, row) in zip(texts, topics_df.iterrows()):
                text.topic = row.values.flatten().tolist()
        else:
            topics = self.config.topic_modeler.fit_transform(texts=texts)
            for text, topic in zip(texts, topics):
                text.topic = topic
        return texts