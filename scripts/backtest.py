#!/usr/bin/env python
"""Standalone backtest script. Run: python scripts/backtest.py BTCUSDT"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.database import init_db, SessionLocal, Asset
from src.models.trainer import Trainer
from src.models.backtester import Backtester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/backtest.py <SYMBOL>")
        print("Example: python scripts/backtest.py BTCUSDT")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    init_db()

    # Load model
    logger.info(f"Loading model for {symbol}...")
    model = Trainer.load_model(symbol)

    # Load scaler
    scaler_path = config.MODEL_DIR / f"{symbol}_scaler.json"
    if not scaler_path.exists():
        logger.error(f"Scaler not found at {scaler_path}")
        sys.exit(1)

    with open(scaler_path) as f:
        scaler_params = json.load(f)

    # Load candles
    logger.info("Loading candle data...")
    trainer = Trainer(symbol)
    candles_by_tf = {}
    for tf in config.ALL_TIMEFRAMES:
        df = trainer.load_candles(tf)
        if df.empty:
            logger.error(f"No data for {symbol} {tf}")
            sys.exit(1)
        candles_by_tf[tf] = df
        logger.info(f"  {tf}: {len(df)} candles")

    # Run backtest
    logger.info("Running backtest...")
    backtester = Backtester(symbol, model)
    result = backtester.run(candles_by_tf, scaler_params)

    # Print results
    print("\n" + "=" * 50)
    print(f"BACKTEST RESULTS — {symbol}")
    print("=" * 50)
    print(f"Total Signals:  {result.total_signals}")
    print(f"Wins:           {result.wins}")
    print(f"Losses:         {result.losses}")
    print(f"Expired:        {result.expired}")
    print(f"Win Rate:       {result.win_rate:.2%}")
    print(f"Profit Factor:  {result.profit_factor:.2f}")
    print(f"Avg R:R:        {result.avg_rr:.2f}")
    print(f"Max Drawdown:   {result.max_drawdown:.2f}R")
    print(f"TP1 Hit Rate:   {result.tp1_hits}/{result.total_signals}")
    print(f"TP2 Hit Rate:   {result.tp2_hits}/{result.total_signals}")
    print(f"TP3 Hit Rate:   {result.tp3_hits}/{result.total_signals}")
    print("-" * 50)
    passed = "PASS" if result.passed else "FAIL"
    print(f"Threshold:      {passed}")
    print(f"  Min WR: {config.MIN_WIN_RATE:.0%} | Min PF: {config.MIN_PROFIT_FACTOR}")
    print("=" * 50)

    # Update DB
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if asset:
            asset.backtest_win_rate = result.win_rate
            asset.backtest_profit_factor = result.profit_factor
            if result.passed:
                asset.status = "ACTIVE"
            else:
                asset.status = "FAILED"
            db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    main()
