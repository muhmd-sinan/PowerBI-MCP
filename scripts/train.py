#!/usr/bin/env python
"""Standalone training script. Run: python scripts/train.py BTCUSDT"""
import sys
import logging
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.data.database import init_db, SessionLocal, Asset
from src.data.historical import fetch_historical
from src.models.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/train.py <SYMBOL> [--fetch]")
        print("Example: python scripts/train.py BTCUSDT --fetch")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    should_fetch = "--fetch" in sys.argv

    init_db()

    if should_fetch:
        logger.info(f"Fetching historical data for {symbol}...")
        for tf in config.ALL_TIMEFRAMES:
            logger.info(f"  Fetching {tf}...")
            await fetch_historical(symbol, tf, months=config.LOOKBACK_MONTHS)
        logger.info("Fetch complete")

    logger.info(f"Training model for {symbol}...")
    trainer = Trainer(symbol)
    result = trainer.train(epochs=100, batch_size=64)
    model_path = trainer.save()

    logger.info(f"Training complete:")
    logger.info(f"  Test accuracy: {result['test_accuracy']:.4f}")
    logger.info(f"  Epochs trained: {result['epochs_trained']}")
    logger.info(f"  Model saved: {model_path}")

    # Update asset record
    from datetime import datetime
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if not asset:
            asset = Asset(symbol=symbol, status="ACTIVE")
            db.add(asset)
        asset.model_path = model_path
        asset.last_trained = datetime.utcnow()
        db.commit()
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
