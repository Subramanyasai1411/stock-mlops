"""Model Registry — versioning, saving, loading."""
import os, json
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, text
from tensorflow.keras.models import load_model, Model
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, database_url: str, model_dir: str):
        self.engine = create_engine(database_url)
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def save_model(self, model: Model, ticker: str, metrics: Dict, model_name="resnet_lstm") -> str:
        version = datetime.now().strftime("v%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, ticker, version)
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, "model.keras")
        model.save(model_file)

        with self.engine.connect() as conn:
            conn.execute(text("UPDATE model_registry SET is_active=FALSE WHERE ticker=:t"), {'t': ticker})
            conn.execute(text("""
                INSERT INTO model_registry (model_name, version, ticker, model_path, metrics, is_active)
                VALUES (:mn, :v, :t, :mp, :m, TRUE)
            """), {'mn': model_name, 'v': version, 't': ticker, 'mp': model_file, 'm': json.dumps(metrics)})
            conn.commit()
        logger.info(f"Saved {ticker} model {version}")
        return version

    def load_model(self, ticker: str, version: Optional[str] = None) -> Model:
        with self.engine.connect() as conn:
            if version:
                r = conn.execute(text("SELECT model_path FROM model_registry WHERE ticker=:t AND version=:v"),
                                 {'t': ticker, 'v': version})
            else:
                r = conn.execute(text(
                    "SELECT model_path FROM model_registry WHERE ticker=:t AND is_active=TRUE ORDER BY created_at DESC LIMIT 1"),
                    {'t': ticker})
            row = r.fetchone()
        if not row:
            raise ValueError(f"No model for {ticker}")
        return load_model(row[0])

    def get_active_model_info(self, ticker: str) -> Optional[Dict]:
        with self.engine.connect() as conn:
            r = conn.execute(text("""
                SELECT model_name, version, model_path, metrics, created_at
                FROM model_registry WHERE ticker=:t AND is_active=TRUE
            """), {'t': ticker})
            row = r.fetchone()
        if not row:
            return None
        metrics = row[3] if isinstance(row[3], dict) else json.loads(row[3]) if row[3] else {}
        return {'model_name': row[0], 'version': row[1], 'model_path': row[2], 'metrics': metrics, 'created_at': str(row[4])}

    def list_models(self, ticker=None):
        with self.engine.connect() as conn:
            if ticker:
                r = conn.execute(text("SELECT model_name,version,ticker,is_active,created_at FROM model_registry WHERE ticker=:t ORDER BY created_at DESC"), {'t': ticker})
            else:
                r = conn.execute(text("SELECT model_name,version,ticker,is_active,created_at FROM model_registry ORDER BY created_at DESC"))
            return [{'model_name':x[0],'version':x[1],'ticker':x[2],'is_active':x[3],'created_at':str(x[4])} for x in r.fetchall()]
