"""Data preprocessing — scaling and sequence creation."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import pickle, os, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, sequence_length: int = 30):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'close_price') -> Tuple[np.ndarray, np.ndarray]:
        if df.empty or len(df) < self.sequence_length + 1:
            raise ValueError(f"Need at least {self.sequence_length + 1} data points, got {len(df)}")
        prices = df[target_col].values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(prices)
        self.is_fitted = True
        X, y = self.create_sequences(scaled)
        logger.info(f"Created {len(X)} sequences")
        return X, y

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data.reshape(-1, 1))

    def train_test_split(self, X, y, ratio=0.8):
        s = int(len(X) * ratio)
        return X[:s], X[s:], y[:s], y[s:]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'sequence_length': self.sequence_length, 'is_fitted': self.is_fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        p = cls(sequence_length=state['sequence_length'])
        p.scaler = state['scaler']
        p.is_fitted = state['is_fitted']
        return p
