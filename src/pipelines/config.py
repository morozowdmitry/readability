from typing import Iterable, List, Set, Tuple, Optional
from pathlib import Path

from src.preprocess.base_preprocessor import BasePreprocessor
from src.feature_extract.base_extractor import BaseExtractor
from src.predict.base_predictor import BasePredictor
from src.label.labels import LabelType
from src.label.base_labeler import BaseLabeler
from src.vectorize.vectorizer import BaseVectorizer
from src.utils.scaler import BaseScaler


class PipelineConfig(object):
    def __init__(
            self,
            preprocessor: BasePreprocessor,
            labelers: List[Tuple[BaseLabeler, Set[LabelType]]],
            extractors: Iterable[BaseExtractor],
            predictor: BasePredictor,
            model_path: Path,
            labeled_path: Optional[Path] = None,
            vectorizer: Optional[BaseVectorizer] = None,
            vectorizer_train_only: Optional[bool] = False,
            topic_modeler: Optional[BaseVectorizer] = None,
            topics_path: Optional[Path] = None,
            scaler: Optional[BaseScaler] = None,
    ):
        self.preprocessor = preprocessor
        self.labelers = labelers
        self.extractors = extractors
        self.predictor = predictor
        self.vectorizer = vectorizer
        self.vectorizer_train_only = vectorizer_train_only
        self.topic_modeler = topic_modeler
        self.scaler = scaler
        self.labeled_path = labeled_path
        self.model_path = model_path
        self.topics_path = topics_path
