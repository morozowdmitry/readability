from typing import Optional
from src.utils.scaler import BaseScaler

import pandas as pd
from pathlib import Path
import pickle


class BasePredictor(object):
    def __init__(self, random_state=0):
        self.random_state = random_state

    def train(self, X: pd.DataFrame, y: pd.Series, model_path: Path, scaler: Optional[BaseScaler] = None):
        if scaler is not None:
            X = scaler.fit_transform(X)

        clf = self._create_model(self.random_state)
        clf.fit(X, y)

        with model_path.open('wb') as f:
            pickle.dump(clf, f)

        return self

    def inference(self, X: pd.DataFrame, model_path: Path, scaler: Optional[BaseScaler] = None):
        if scaler is not None:
            X = scaler.transform(X)

        with model_path.open('rb') as f:
            clf = pickle.load(f)

        return clf.predict(X)

    @staticmethod
    def _create_model(random_state):
        raise NotImplementedError
