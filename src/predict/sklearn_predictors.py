from typing import Optional
from src.utils.scaler import BaseScaler

from pathlib import Path
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from src.predict.base_predictor import BasePredictor


class RandomForestPredictor(BasePredictor):
    @staticmethod
    def _create_model(random_state):
        return RandomForestClassifier(random_state=random_state, verbose=1)


class SVCPredictor(BasePredictor):
    @staticmethod
    def _create_model(random_state):
        return LinearSVC(random_state=random_state, verbose=1)
