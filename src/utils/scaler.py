from pathlib import Path
import pandas as pd

import pickle
from sklearn.preprocessing import MinMaxScaler


class BaseScaler:
    def __init__(self, model_path: Path):
        self.path = model_path

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError


class SKLearnMinMaxScaler(BaseScaler):
    def fit_transform(self, df: pd.DataFrame):
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df)
        self.path.parent.mkdir(exist_ok=True)
        with self.path.open('wb') as f:
            pickle.dump(scaler, f)
        return pd.DataFrame(normalized, columns=df.columns)

    def transform(self, df: pd.DataFrame):
        with self.path.open('rb') as f:
            scaler = pickle.load(f)
        normalized = scaler.transform(df)
        return pd.DataFrame(normalized, columns=df.columns)

