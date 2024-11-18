from typing import List, Optional

from src.pipelines.base_pipeline import BasePipeline
from src.pipelines.exceptions import NoDataError
from src.convertor.convert2df import texts2df


class LabelPipeline(BasePipeline):
    def run(self, raw_texts: List[str] = None):
        if raw_texts is None:
            raise NoDataError
        labeled, prelabeled_df = self._prepare_data(raw_texts=raw_texts)
        labeled_df = texts2df(labeled)
        labeled_df.to_csv(self.config.labeled_path, index=False)
