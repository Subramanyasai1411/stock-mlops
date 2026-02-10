"""Inference — single and multi-day predictions with yfinance fallback."""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
import logging

from ..models.registry import ModelRegistry
from ..data.ingestion import StockDataIngestion
from ..data.preprocessing import DataPreprocessor

logger = logging.getLogger(__name__)


class StockPredictor:
    def __init__(self, database_url, model_dir, sequence_length=30):
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.ingestion = StockDataIngestion(database_url)
        self.registry = ModelRegistry(database_url, model_dir)
        self._model_cache = {}
        self._preprocessor_cache = {}

    def _load(self, ticker, version=None):
        key = f"{ticker}_{version or 'active'}"
        if key not in self._model_cache:
            info = self.registry.get_active_model_info(ticker)
            if not info:
                raise ValueError(f"No trained model for {ticker}")
            ver = version or info['version']
            model_path = info.get('model_path', '')
            if not os.path.exists(model_path):
                raise ValueError(f"Model file missing: {model_path}")
            model = self.registry.load_model(ticker, version)
            pp_path = os.path.join(self.model_dir, ticker, ver, "preprocessor.pkl")
            if not os.path.exists(pp_path):
                raise ValueError(f"Preprocessor missing: {pp_path}")
            pp = DataPreprocessor.load(pp_path)
            self._model_cache[key] = model
            self._preprocessor_cache[key] = pp
        return self._model_cache[key], self._preprocessor_cache[key]

    def predict(self, ticker, days=1, version=None) -> Dict[str, Any]:
        days = max(1, min(30, days))
        try:
            model, pp = self._load(ticker, version)

            # Get data from DB
            df = self.ingestion.get_historical_data(ticker, days=self.sequence_length * 5)
            # Fallback to yfinance if insufficient
            if len(df) < self.sequence_length:
                logger.warning(f"{ticker}: Only {len(df)} rows in DB, fetching from yfinance")
                df = self.ingestion.fetch_direct(ticker, days=self.sequence_length * 3)
            if len(df) < self.sequence_length:
                raise ValueError(f"Need {self.sequence_length} data points, got {len(df)}")

            prices = df['close_price'].values[-self.sequence_length:].reshape(-1, 1)
            scaled = pp.scaler.transform(prices).flatten()

            preds = []
            seq = scaled.copy()
            for _ in range(days):
                inp = seq.reshape(1, self.sequence_length, 1)
                p = model.predict(inp, verbose=0)[0][0]
                preds.append(p)
                seq = np.append(seq[1:], p)

            prices_out = pp.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

            info = self.registry.get_active_model_info(ticker)
            ver = info['version'] if info else (version or 'unknown')

            last_date = df['date'].iloc[-1]
            if isinstance(last_date, str):
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

            forecast = []
            d = last_date
            for i, price in enumerate(prices_out):
                d = d + timedelta(days=1)
                while d.weekday() >= 5:
                    d = d + timedelta(days=1)
                forecast.append({'day': i+1, 'date': str(d), 'predicted_price': round(float(price), 2)})

            try:
                with self.engine.connect() as conn:
                    conn.execute(text("""INSERT INTO predictions (ticker,prediction_date,predicted_price,model_version)
                        VALUES (:t,:d,:p,:v) ON CONFLICT DO NOTHING"""),
                        {'t': ticker, 'd': datetime.now().date(), 'p': float(prices_out[0]), 'v': ver})
                    conn.commit()
            except:
                pass

            return {
                'ticker': ticker, 'days': days, 'forecast': forecast,
                'predicted_price': round(float(prices_out[0]), 2),
                'prediction_date': forecast[0]['date'], 'model_version': ver,
                'last_actual_price': round(float(df['close_price'].iloc[-1]), 2),
                'last_date': str(last_date)
            }
        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}", exc_info=True)
            return {'ticker': ticker, 'error': str(e), 'status': 'failed'}

    def predict_all(self, tickers, days=1):
        return {t: self.predict(t, days=days) for t in tickers}

    def clear_cache(self):
        self._model_cache.clear()
        self._preprocessor_cache.clear()

    def get_prediction_history(self, ticker, days=30):
        with self.engine.connect() as conn:
            r = conn.execute(text("""SELECT prediction_date,predicted_price,actual_price,model_version
                FROM predictions WHERE ticker=:t AND prediction_date>=CURRENT_DATE-CAST(:d AS INTEGER)
                ORDER BY prediction_date DESC"""), {'t': ticker, 'd': days})
            return [{'date':str(x[0]),'predicted':float(x[1]),'actual':float(x[2]) if x[2] else None,'model_version':x[3]} for x in r.fetchall()]
