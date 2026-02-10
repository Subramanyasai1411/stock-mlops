import os
from dataclasses import dataclass, field

@dataclass
class Config:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://mlops:mlops123@localhost:5432/mlops")
    model_dir: str = os.getenv("MODEL_DIR", "./models")
    data_dir: str = os.getenv("DATA_DIR", "./data")
    sequence_length: int = int(os.getenv("SEQUENCE_LENGTH", "30"))
    train_epochs: int = int(os.getenv("TRAIN_EPOCHS", "50"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    learning_rate: float = 0.001
    stock_tickers: list = field(default_factory=list)

    def __post_init__(self):
        self.stock_tickers = [t.strip() for t in os.getenv("STOCK_TICKERS", "AAPL,GOOGL,MSFT").split(",")]
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

config = Config()
