"""Training pipeline."""
import os, math
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error
import logging

from ..models.resnet_lstm import build_resnet_lstm
from ..models.registry import ModelRegistry
from ..data.ingestion import StockDataIngestion
from ..data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, database_url, model_dir, sequence_length=30, epochs=50, batch_size=32, learning_rate=0.001):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ingestion = StockDataIngestion(database_url)
        self.registry = ModelRegistry(database_url, model_dir)

    def train(self, ticker: str, days_of_data=730) -> Dict[str, Any]:
        logger.info(f"Training {ticker}...")
        try:
            df = self.ingestion.get_historical_data(ticker, days=days_of_data)
            # Fallback to yfinance if DB is empty
            if len(df) < self.sequence_length + 10:
                logger.warning(f"{ticker}: Only {len(df)} rows in DB, fetching from yfinance...")
                df = self.ingestion.fetch_direct(ticker, days=days_of_data)
            if len(df) < self.sequence_length + 10:
                raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")

            preprocessor = DataPreprocessor(self.sequence_length)
            X, y = preprocessor.fit_transform(df)
            X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

            model = build_resnet_lstm(self.sequence_length, learning_rate=self.learning_rate)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=self.epochs, batch_size=self.batch_size, verbose=1)

            y_pred = model.predict(X_test)
            y_test_inv = preprocessor.inverse_transform(y_test)
            y_pred_inv = preprocessor.inverse_transform(y_pred)
            rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

            metrics = {'train_loss': float(history.history['loss'][-1]),
                       'val_loss': float(history.history['val_loss'][-1]),
                       'rmse': float(rmse), 'epochs': self.epochs}

            version = self.registry.save_model(model, ticker, metrics)
            preprocessor.save(os.path.join(self.model_dir, ticker, version, "preprocessor.pkl"))

            logger.info(f"{ticker} done — RMSE: {rmse:.4f}")
            return {'status': 'success', 'ticker': ticker, 'version': version, 'metrics': metrics}
        except Exception as e:
            logger.error(f"Training failed for {ticker}: {e}")
            return {'status': 'failed', 'ticker': ticker, 'error': str(e)}

    def train_all(self, tickers, days_of_data=730):
        return {t: self.train(t, days_of_data) for t in tickers}
