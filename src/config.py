import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    BASE_DIR = Path(__file__).parent.parent

    # Binance
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # MySQL
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "futures_bot")

    @property
    def database_url(self):
        return (
            f"mysql+mysqldb://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        )

    # Signal Settings
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.70"))
    ATR_TP1_MULT = float(os.getenv("ATR_TP1_MULT", "1.0"))
    ATR_TP2_MULT = float(os.getenv("ATR_TP2_MULT", "2.0"))
    ATR_TP3_MULT = float(os.getenv("ATR_TP3_MULT", "3.0"))
    ATR_SL_MULT = float(os.getenv("ATR_SL_MULT", "1.5"))

    # Multi-timeframe
    PRIMARY_TIMEFRAMES = ["5m", "10m", "15m"]
    CONFLUENCE_TIMEFRAMES = ["30m", "1h", "2h"]
    ALL_TIMEFRAMES = PRIMARY_TIMEFRAMES + CONFLUENCE_TIMEFRAMES
    PRIMARY_VOTE_THRESHOLD = 2  # 2 out of 3 must agree
    TF_BOOST_MULTIPLIER = 1.2
    TF_PENALTY_MULTIPLIER = 0.7

    # Training
    RETRAIN_DAY = os.getenv("RETRAIN_DAY", "monday")
    RETRAIN_HOUR = int(os.getenv("RETRAIN_HOUR", "3"))
    LOOKBACK_MONTHS = int(os.getenv("LOOKBACK_MONTHS", "12"))
    FORWARD_WINDOW_CANDLES = int(os.getenv("FORWARD_WINDOW_CANDLES", "50"))

    # Backtesting thresholds
    MIN_WIN_RATE = 0.55
    MIN_PROFIT_FACTOR = 1.5
    DEGRADATION_THRESHOLD = 0.50
    DEGRADATION_WINDOW = 30  # last N signals

    # Web
    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("WEB_PORT", "8000"))

    # Assets
    DEFAULT_ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Model
    MODEL_DIR = BASE_DIR / "models"
    MODEL_DIR.mkdir(exist_ok=True)


config = Config()
