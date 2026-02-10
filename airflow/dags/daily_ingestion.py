import os, sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

default_args = {'owner': 'mlops', 'retries': 2, 'retry_delay': timedelta(minutes=5)}

def ingest(**ctx):
    from src.data.ingestion import StockDataIngestion
    ing = StockDataIngestion(os.getenv('DATABASE_URL'))
    tickers = os.getenv('STOCK_TICKERS', 'AAPL,GOOGL,MSFT').split(',')
    print(ing.ingest(tickers, days_back=7))

with DAG('daily_stock_ingestion', default_args=default_args, schedule_interval='0 18 * * 1-5',
         start_date=datetime(2024,1,1), catchup=False, tags=['data']) as dag:
    PythonOperator(task_id='ingest', python_callable=ingest)
