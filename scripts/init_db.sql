CREATE DATABASE IF NOT EXISTS futures_bot CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE futures_bot;

-- Tables are created by SQLAlchemy ORM (src/data/database.py)
-- Run: python -c "from src.data.database import init_db; init_db()"

-- Manual creation for reference:

CREATE TABLE IF NOT EXISTS candles (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    timestamp BIGINT NOT NULL,
    `open` DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    `close` DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    UNIQUE KEY uq_candle (symbol, timeframe, timestamp),
    INDEX ix_candle_lookup (symbol, timeframe, timestamp)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS signals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction ENUM('LONG', 'SHORT') NOT NULL,
    entry_price DOUBLE NOT NULL,
    tp1 DOUBLE NOT NULL,
    tp2 DOUBLE NOT NULL,
    tp3 DOUBLE NOT NULL,
    sl DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    status ENUM('ACTIVE', 'WON', 'LOST', 'EXPIRED') NOT NULL DEFAULT 'ACTIVE',
    tp1_hit BOOLEAN DEFAULT FALSE,
    tp2_hit BOOLEAN DEFAULT FALSE,
    tp3_hit BOOLEAN DEFAULT FALSE,
    sl_hit BOOLEAN DEFAULT FALSE,
    candles_elapsed INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    resolved_at DATETIME NULL,
    INDEX ix_signal_symbol_status (symbol, status)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS assets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    status ENUM('ACTIVE', 'FETCHING', 'TRAINING', 'BACKTESTING', 'FAILED', 'INACTIVE') NOT NULL DEFAULT 'INACTIVE',
    model_path VARCHAR(255) NULL,
    last_trained DATETIME NULL,
    backtest_win_rate DOUBLE NULL,
    backtest_profit_factor DOUBLE NULL,
    live_win_rate DOUBLE NULL,
    total_signals INT DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS config (
    `key` VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;
