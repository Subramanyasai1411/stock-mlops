"""Data ingestion — fetches stock data from Yahoo Finance and stores in PostgreSQL."""
import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard requests session (no curl_cffi)
_SESSION = requests.Session()
_SESSION.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})


def _fetch_yahoo_raw(ticker: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """
    Fallback: Fetch directly from Yahoo Finance v8 API using raw HTTP.
    This bypasses yfinance entirely.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d',
        'events': 'history',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }

    try:
        resp = _SESSION.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        result = data.get('chart', {}).get('result', [])
        if not result:
            logger.warning(f"Yahoo raw API: No results for {ticker}")
            return pd.DataFrame()

        chart = result[0]
        timestamps = chart.get('timestamp', [])
        quote = chart.get('indicators', {}).get('quote', [{}])[0]

        if not timestamps:
            return pd.DataFrame()

        df = pd.DataFrame({
            'date': [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
            'open_price': quote.get('open', []),
            'high_price': quote.get('high', []),
            'low_price': quote.get('low', []),
            'close_price': quote.get('close', []),
            'volume': quote.get('volume', []),
        })
        df['ticker'] = ticker
        df['date'] = pd.to_datetime(df['date']).dt.date
        # Drop rows with NaN prices
        df = df.dropna(subset=['close_price'])
        logger.info(f"Yahoo raw API: Got {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Yahoo raw API failed for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_yfinance_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Method 1: Use yf.Ticker().history() with requests session.
    This often works when yf.download() doesn't.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker, session=_SESSION)
        df = t.history(start=start, end=end, auto_adjust=True)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        # Handle timezone-aware datetime
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)

        df['Ticker'] = ticker
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open_price', 'High': 'high_price',
            'Low': 'low_price', 'Close': 'close_price', 'Volume': 'volume',
            'Ticker': 'ticker'
        })
        df['date'] = pd.to_datetime(df['date']).dt.date
        cols = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        logger.info(f"yf.Ticker.history: Got {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.warning(f"yf.Ticker.history failed for {ticker}: {e}")
        return pd.DataFrame()


def _fetch_yfinance_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Method 2: Use yf.download() with requests session.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False,
                         auto_adjust=True, session=_SESSION)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[1] == '' or col[1] == ticker else col[0] for col in df.columns]

        df['Ticker'] = ticker
        df = df.rename(columns={
            'Date': 'date', 'Open': 'open_price', 'High': 'high_price',
            'Low': 'low_price', 'Close': 'close_price', 'Volume': 'volume',
            'Ticker': 'ticker'
        })
        df['date'] = pd.to_datetime(df['date']).dt.date
        cols = ['ticker', 'date', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        logger.info(f"yf.download: Got {len(df)} rows for {ticker}")
        return df

    except Exception as e:
        logger.warning(f"yf.download failed for {ticker}: {e}")
        return pd.DataFrame()


class StockDataIngestion:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url)

    def fetch_stock_data(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch stock data trying 3 methods in order."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        # Method 1: yf.Ticker().history()
        df = _fetch_yfinance_ticker(ticker, start_date, end_date)
        if not df.empty:
            return df

        # Method 2: yf.download()
        logger.info(f"  Trying yf.download for {ticker}...")
        df = _fetch_yfinance_download(ticker, start_date, end_date)
        if not df.empty:
            return df

        # Method 3: Raw Yahoo Finance API
        logger.info(f"  Trying raw Yahoo API for {ticker}...")
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        df = _fetch_yahoo_raw(ticker, start_ts, end_ts)
        if not df.empty:
            return df

        logger.error(f"ALL 3 methods failed for {ticker}")
        return pd.DataFrame()

    def save_to_database(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        count = 0
        with self.engine.connect() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute(text("""
                        INSERT INTO stock_prices (ticker, date, open_price, high_price, low_price, close_price, volume)
                        VALUES (:ticker, :date, :open_price, :high_price, :low_price, :close_price, :volume)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open_price=EXCLUDED.open_price, high_price=EXCLUDED.high_price,
                            low_price=EXCLUDED.low_price, close_price=EXCLUDED.close_price, volume=EXCLUDED.volume
                    """), {
                        'ticker': row['ticker'], 'date': row['date'],
                        'open_price': float(row['open_price']), 'high_price': float(row['high_price']),
                        'low_price': float(row['low_price']), 'close_price': float(row['close_price']),
                        'volume': int(row['volume'])
                    })
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to save row: {e}")
            conn.commit()
        logger.info(f"Saved {count} records to database")
        return count

    def ingest(self, tickers: List[str], days_back: int = 730) -> dict:
        results = {}
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        for ticker in tickers:
            try:
                df = self.fetch_stock_data(ticker, start_date=start)
                count = self.save_to_database(df)
                results[ticker] = {"status": "success", "records": count}
            except Exception as e:
                logger.error(f"Error ingesting {ticker}: {e}")
                results[ticker] = {"status": "error", "error": str(e)}
            time.sleep(1)  # Rate limit between tickers
        return results

    def get_historical_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        query = text("""
            SELECT date, open_price, high_price, low_price, close_price, volume
            FROM stock_prices
            WHERE ticker = :ticker AND date >= CURRENT_DATE - CAST(:days AS INTEGER)
            ORDER BY date ASC
        """)
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={'ticker': ticker, 'days': days})
        return df

    def fetch_direct(self, ticker: str, days: int = 120) -> pd.DataFrame:
        """Fetch directly from yfinance without DB — fallback for predictions."""
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        return self.fetch_stock_data(ticker, start_date=start, end_date=end)