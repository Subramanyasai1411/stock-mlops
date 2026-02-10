"""FastAPI — Stock Prediction MLOps API."""
import os, sys, threading
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

sys.path.insert(0, '/app')

from src.inference.predictor import StockPredictor
from src.models.registry import ModelRegistry
from src.data.ingestion import StockDataIngestion
from src.training.trainer import ModelTrainer

DATABASE_URL = os.getenv('DATABASE_URL')
MODEL_DIR = os.getenv('MODEL_DIR', '/app/models')
SEQ_LEN = int(os.getenv('SEQUENCE_LENGTH', '30'))
TICKERS = [t.strip() for t in os.getenv('STOCK_TICKERS', 'AAPL,GOOGL,MSFT').split(',')]
EPOCHS = int(os.getenv('TRAIN_EPOCHS', '50'))
BATCH = int(os.getenv('BATCH_SIZE', '32'))

predictor = StockPredictor(DATABASE_URL, MODEL_DIR, SEQ_LEN)
registry = ModelRegistry(DATABASE_URL, MODEL_DIR)
ingestion = StockDataIngestion(DATABASE_URL)
trainer = ModelTrainer(DATABASE_URL, MODEL_DIR, SEQ_LEN, EPOCHS, BATCH)

training_status = {"is_training": False, "current_ticker": None, "progress": {}, "last_trained": None}

app = FastAPI(title="Stock Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ── Startup ──
@app.on_event("startup")
async def startup():
    def _go():
        global training_status
        try:
            print("=" * 60)
            print("STARTUP: Ingesting stock data...")
            print("=" * 60)
            results = ingestion.ingest(TICKERS, days_back=730)
            for t, r in results.items():
                print(f"  {t}: {r}")

            needs = []
            for t in TICKERS:
                info = registry.get_active_model_info(t)
                if info and os.path.exists(info.get('model_path', '')):
                    print(f"  {t}: Model OK ({info['version']})")
                else:
                    needs.append(t)

            if needs:
                print("=" * 60)
                print(f"STARTUP: Training {needs}...")
                print("=" * 60)
                training_status["is_training"] = True
                for t in needs:
                    training_status["current_ticker"] = t
                    r = trainer.train(t, days_of_data=730)
                    training_status["progress"][t] = r
                    s = r.get('status')
                    print(f"  {t}: {'RMSE='+str(round(r['metrics']['rmse'],4)) if s=='success' else 'FAILED - '+r.get('error','')}")
                training_status["is_training"] = False
                training_status["current_ticker"] = None
                training_status["last_trained"] = datetime.now().isoformat()
            print("=" * 60)
            print("STARTUP COMPLETE")
            print("=" * 60)
        except Exception as e:
            print(f"STARTUP ERROR: {e}")
            training_status["is_training"] = False
    threading.Thread(target=_go, daemon=True).start()


# ── Retrain ──
def _retrain():
    global training_status
    training_status.update({"is_training": True, "progress": {}})
    try:
        ingestion.ingest(TICKERS, days_back=730)
        for t in TICKERS:
            training_status["current_ticker"] = t
            r = trainer.train(t, days_of_data=730)
            training_status["progress"][t] = r
        predictor.clear_cache()
        training_status["last_trained"] = datetime.now().isoformat()
    except Exception as e:
        print(f"RETRAIN ERROR: {e}")
    finally:
        training_status.update({"is_training": False, "current_ticker": None})

@app.post("/retrain")
async def retrain():
    if training_status["is_training"]:
        return {"status": "already_training", "current_ticker": training_status["current_ticker"]}
    threading.Thread(target=_retrain, daemon=True).start()
    return {"status": "started", "tickers": TICKERS}

@app.get("/training-status")
async def get_training_status():
    return training_status


# ── Dashboard ──
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard")
async def dashboard():
    return FileResponse('/app/dashboard/index.html')


# ── Predictions ──
@app.get("/predict/{ticker}")
async def predict(ticker: str, days: int = 1, version: Optional[str] = None):
    if training_status["is_training"]:
        return {"ticker": ticker.upper(), "error": "Models are training, please wait..."}
    return predictor.predict(ticker.upper(), days=days, version=version)

@app.get("/predict")
async def predict_multi(tickers: str, days: int = 1):
    return [predictor.predict(t.strip().upper(), days=days) for t in tickers.split(',')]


# ── Models ──
@app.get("/models")
async def list_models(ticker: Optional[str] = None):
    return registry.list_models(ticker.upper() if ticker else None)

@app.get("/models/{ticker}/active")
async def active_model(ticker: str):
    info = registry.get_active_model_info(ticker.upper())
    if not info:
        raise HTTPException(404, f"No active model for {ticker}")
    return info


# ── Data ──
@app.get("/data/{ticker}")
async def stock_data(ticker: str, days: int = 365):
    df = ingestion.get_historical_data(ticker.upper(), days)
    if df.empty:
        df = ingestion.fetch_direct(ticker.upper(), days)
    return df.to_dict(orient='records')

@app.post("/cache/clear")
async def clear_cache():
    predictor.clear_cache()
    return {"status": "cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
