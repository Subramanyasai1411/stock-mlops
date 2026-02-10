# 📈 Stock MLOps — ResNet-LSTM Forecasting Pipeline

A production-grade MLOps pipeline for stock price forecasting using a hybrid **ResNet-LSTM** deep learning architecture, containerized with Docker and orchestrated with Apache Airflow.

**🔗 [Live Dashboard Demo](https://YOUR_USERNAME.github.io/stock-mlops/)**

---

## Architecture

```
Yahoo Finance ──→ Data Ingestion ──→ PostgreSQL
                                         │
                                    Preprocessing
                                   (MinMaxScaler + Sequences)
                                         │
                                    ResNet-LSTM Model
                                   (Conv1D → ResBlocks → LSTM → Dense)
                                         │
                               ┌─────────┴─────────┐
                          Model Registry        FastAPI Server
                         (Versioned Storage)    (/predict, /data, /models)
                               │                     │
                          Airflow DAGs          Dashboard UI
                        (Daily Ingest,        (Chart.js + Real-time)
                         Monthly Retrain)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Model** | TensorFlow/Keras — ResNet-LSTM hybrid (Conv1D + Residual Blocks + LSTM) |
| **API** | FastAPI with CORS, async startup, background training |
| **Database** | PostgreSQL 15 — stock prices, model registry, predictions |
| **Orchestration** | Apache Airflow — daily ingestion + monthly retraining DAGs |
| **Containerization** | Docker Compose — 3-service architecture (API, DB, Airflow) |
| **Data** | Yahoo Finance (3-method fallback: yf.Ticker, yf.download, raw API) |
| **Frontend** | Vanilla JS + Chart.js — real-time dashboard with multi-day forecasting |

## Model Architecture

```
Input (30 timesteps × 1 feature)
    ↓
Conv1D(64, kernel=3) → BatchNorm → ReLU
    ↓
ResNet Block ×2
    ├── Conv1D → BatchNorm → ReLU
    ├── Conv1D → BatchNorm
    ├── + Shortcut Connection
    └── ReLU
    ↓
LSTM(64 units)
    ↓
Dropout(0.2)
    ↓
Dense(1) → Predicted Price
```

**Optimizer:** Adam (lr=0.001) · **Loss:** MSE · **Metrics:** MAE, RMSE

## Project Structure

```
stock-mlops/
├── src/
│   ├── data/
│   │   ├── ingestion.py          # 3-method Yahoo Finance fallback
│   │   └── preprocessing.py      # MinMaxScaler + sequence creation
│   ├── models/
│   │   ├── resnet_lstm.py         # Hybrid architecture definition
│   │   └── registry.py            # Versioned model storage & tracking
│   ├── training/
│   │   └── trainer.py             # End-to-end training pipeline
│   └── inference/
│       └── predictor.py           # Multi-day autoregressive prediction
├── api/
│   ├── main.py                    # FastAPI server with all endpoints
│   ├── Dockerfile
│   └── requirements.txt
├── airflow/
│   ├── dags/
│   │   ├── daily_ingestion.py     # Weekday 6PM data fetch
│   │   └── monthly_retrain.py     # 1st-of-month model retraining
│   ├── Dockerfile
│   └── requirements.txt
├── database/
│   └── init.sql                   # Schema: stock_prices, model_registry, predictions
├── dashboard/
│   └── index.html                 # Interactive Chart.js dashboard
├── models/                        # Versioned model artifacts (.keras + preprocessor.pkl)
├── docker-compose.yml
├── .env
└── README.md
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- 4GB+ RAM (for TensorFlow training)

### Run

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/stock-mlops.git
cd stock-mlops

# Start all services
docker-compose up --build -d

# Dashboard: http://localhost:8000
# API docs:  http://localhost:8000/docs
# Airflow:   http://localhost:8080 (admin/admin)
```

On first startup, the API automatically:
1. Ingests 2 years of stock data from Yahoo Finance
2. Trains ResNet-LSTM models for each configured ticker (AAPL, GOOGL, MSFT)
3. Registers trained models in PostgreSQL with version tracking

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/predict/{ticker}?days=7` | Multi-day forecast (1–30 days) |
| `GET` | `/data/{ticker}?days=365` | Historical price data |
| `GET` | `/models/{ticker}/active` | Active model info + metrics |
| `POST` | `/retrain` | Trigger full retraining pipeline |
| `GET` | `/training-status` | Live training progress |
| `GET` | `/dashboard` | Interactive web dashboard |

### Configuration

Edit `.env` to customize:

```env
STOCK_TICKERS=AAPL,GOOGL,MSFT      # Tickers to track
SEQUENCE_LENGTH=30                    # Lookback window
TRAIN_EPOCHS=50                       # Training epochs
BATCH_SIZE=32                         # Batch size
```

## Key Features

- **3-Method Data Fallback** — `yf.Ticker.history()` → `yf.download()` → raw Yahoo Finance API, ensuring reliable data ingestion even when individual methods fail
- **Model Versioning** — Every trained model is versioned (`v{YYYYMMDD_HHMMSS}`), stored with its preprocessor, and tracked in PostgreSQL with automatic active model switching
- **Multi-Day Forecasting** — Autoregressive prediction supporting 1–30 day horizons with business-day aware date generation
- **Background Training** — Non-blocking model training via threading, with real-time progress tracking through the API and dashboard
- **Airflow Orchestration** — Automated daily data ingestion (weekdays 6PM) and monthly model retraining with full DAG visibility
