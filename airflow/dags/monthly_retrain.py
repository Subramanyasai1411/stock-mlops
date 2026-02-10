import os, sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

default_args = {'owner': 'mlops', 'retries': 1, 'retry_delay': timedelta(minutes=10)}

def retrain(**ctx):
    from src.training.trainer import ModelTrainer
    t = ModelTrainer(os.getenv('DATABASE_URL'), os.getenv('MODEL_DIR', '/app/models'),
                     int(os.getenv('SEQUENCE_LENGTH','30')), int(os.getenv('TRAIN_EPOCHS','50')), int(os.getenv('BATCH_SIZE','32')))
    tickers = os.getenv('STOCK_TICKERS', 'AAPL,GOOGL,MSFT').split(',')
    print(t.train_all(tickers))

with DAG('monthly_retrain', default_args=default_args, schedule_interval='0 2 1 * *',
         start_date=datetime(2024,1,1), catchup=False, tags=['training']) as dag:
    PythonOperator(task_id='retrain', python_callable=retrain, execution_timeout=timedelta(hours=4))
