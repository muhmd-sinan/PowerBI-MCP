from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Enum, Text,
    BigInteger, UniqueConstraint, Index, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from src.config import config

Base = declarative_base()


class Candle(Base):
    __tablename__ = "candles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(5), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("ix_candle_lookup", "symbol", "timeframe", "timestamp"),
    )


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(Enum("LONG", "SHORT"), nullable=False)
    entry_price = Column(Float, nullable=False)
    tp1 = Column(Float, nullable=False)
    tp2 = Column(Float, nullable=False)
    tp3 = Column(Float, nullable=False)
    sl = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    status = Column(
        Enum("ACTIVE", "WON", "LOST", "EXPIRED"),
        nullable=False,
        default="ACTIVE",
    )
    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    sl_hit = Column(Boolean, default=False)
    candles_elapsed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_signal_symbol_status", "symbol", "status"),
    )


class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), unique=True, nullable=False)
    status = Column(
        Enum("ACTIVE", "FETCHING", "TRAINING", "BACKTESTING", "FAILED", "INACTIVE"),
        nullable=False,
        default="INACTIVE",
    )
    model_path = Column(String(255), nullable=True)
    last_trained = Column(DateTime, nullable=True)
    backtest_win_rate = Column(Float, nullable=True)
    backtest_profit_factor = Column(Float, nullable=True)
    live_win_rate = Column(Float, nullable=True)
    total_signals = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConfigEntry(Base):
    __tablename__ = "config"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


engine = create_engine(config.database_url, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
