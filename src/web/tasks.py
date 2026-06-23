import asyncio
import logging
from datetime import datetime

from src.config import config
from src.data.database import SessionLocal, Asset
from src.data.historical import fetch_historical
from src.models.trainer import Trainer
from src.models.backtester import Backtester

logger = logging.getLogger(__name__)


async def _validate_binance_symbol(symbol: str) -> bool:
    """Check if a symbol exists on Binance Futures."""
    try:
        from binance import AsyncClient

        client = await AsyncClient.create(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
        )
        try:
            info = await client.futures_exchange_info()
            symbols = [s["symbol"] for s in info["symbols"]]
            return symbol in symbols
        finally:
            await client.close_connection()
    except Exception as e:
        logger.error(f"Binance validation failed for {symbol}: {e}")
        return False


def _update_asset_status(symbol: str, status: str, **kwargs):
    """Update asset status in the database."""
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if asset:
            asset.status = status
            for key, value in kwargs.items():
                if hasattr(asset, key):
                    setattr(asset, key, value)
            db.commit()
    finally:
        db.close()


async def add_coin_pipeline(symbol: str):
    """
    Full pipeline for adding a new coin:
    1. Validate on Binance Futures
    2. Fetch 12mo historical data for all timeframes
    3. Train model
    4. Backtest
    5. Activate or mark failed
    """
    logger.info(f"[Pipeline] Starting add_coin_pipeline for {symbol}")

    # Step 1: Validate symbol
    valid = await _validate_binance_symbol(symbol)
    if not valid:
        logger.error(f"[Pipeline] {symbol} not found on Binance Futures")
        _update_asset_status(symbol, "FAILED")
        return

    # Step 2: Fetch historical data
    logger.info(f"[Pipeline] Fetching historical data for {symbol}")
    try:
        for tf in config.ALL_TIMEFRAMES:
            logger.info(f"[Pipeline] Fetching {symbol} {tf}...")
            await fetch_historical(symbol, tf, months=config.LOOKBACK_MONTHS)
        logger.info(f"[Pipeline] Historical data fetch complete for {symbol}")
    except Exception as e:
        logger.error(f"[Pipeline] Historical fetch failed for {symbol}: {e}")
        _update_asset_status(symbol, "FAILED")
        return

    # Step 3: Train model
    _update_asset_status(symbol, "TRAINING")
    logger.info(f"[Pipeline] Training model for {symbol}")
    try:
        trainer = Trainer(symbol)
        trainer.prepare_data()
        trainer.train()
        model_path = trainer.save()
        logger.info(f"[Pipeline] Model trained and saved: {model_path}")
    except Exception as e:
        logger.error(f"[Pipeline] Training failed for {symbol}: {e}")
        _update_asset_status(symbol, "FAILED")
        return

    # Step 4: Backtest
    _update_asset_status(symbol, "BACKTESTING")
    logger.info(f"[Pipeline] Running backtest for {symbol}")
    try:
        backtester = Backtester(symbol, trainer.model)
        # Load candle data for backtest
        import pandas as pd
        from src.data.database import Candle

        db = SessionLocal()
        candles_by_tf = {}
        try:
            for tf in config.ALL_TIMEFRAMES:
                candles = (
                    db.query(Candle)
                    .filter(Candle.symbol == symbol, Candle.timeframe == tf)
                    .order_by(Candle.timestamp.asc())
                    .all()
                )
                candles_by_tf[tf] = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                } for c in candles])
        finally:
            db.close()

        result = backtester.run(candles_by_tf, trainer.scalers)
        logger.info(
            f"[Pipeline] Backtest result for {symbol}: "
            f"WR={result.win_rate:.2%}, PF={result.profit_factor:.2f}, "
            f"Passed={result.passed}"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Backtest failed for {symbol}: {e}")
        _update_asset_status(symbol, "FAILED")
        return

    # Step 5: Activate or fail
    if result.passed:
        _update_asset_status(
            symbol,
            "ACTIVE",
            model_path=model_path,
            last_trained=datetime.utcnow(),
            backtest_win_rate=result.win_rate,
            backtest_profit_factor=result.profit_factor,
        )
        logger.info(f"[Pipeline] {symbol} activated successfully")
    else:
        _update_asset_status(
            symbol,
            "FAILED",
            backtest_win_rate=result.win_rate,
            backtest_profit_factor=result.profit_factor,
        )
        logger.warning(
            f"[Pipeline] {symbol} failed backtest thresholds "
            f"(WR={result.win_rate:.2%} < {config.MIN_WIN_RATE}, "
            f"PF={result.profit_factor:.2f} < {config.MIN_PROFIT_FACTOR})"
        )


async def retrain_asset(symbol: str):
    """
    Retrain an existing asset's model:
    1. Fetch latest data
    2. Train new model
    3. Backtest new model
    4. Compare with existing — swap if better, rollback if worse
    """
    logger.info(f"[Retrain] Starting retrain for {symbol}")

    # Fetch latest data
    try:
        for tf in config.ALL_TIMEFRAMES:
            await fetch_historical(symbol, tf, months=config.LOOKBACK_MONTHS)
    except Exception as e:
        logger.error(f"[Retrain] Data fetch failed for {symbol}: {e}")
        _update_asset_status(symbol, "ACTIVE")  # Revert to active
        return

    # Train new model
    try:
        trainer = Trainer(symbol)
        trainer.prepare_data()
        trainer.train()
    except Exception as e:
        logger.error(f"[Retrain] Training failed for {symbol}: {e}")
        _update_asset_status(symbol, "ACTIVE")
        return

    # Backtest new model
    _update_asset_status(symbol, "BACKTESTING")
    try:
        import pandas as pd
        from src.data.database import Candle

        backtester = Backtester(symbol, trainer.model)
        db = SessionLocal()
        candles_by_tf = {}
        try:
            for tf in config.ALL_TIMEFRAMES:
                candles = (
                    db.query(Candle)
                    .filter(Candle.symbol == symbol, Candle.timeframe == tf)
                    .order_by(Candle.timestamp.asc())
                    .all()
                )
                candles_by_tf[tf] = pd.DataFrame([{
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                } for c in candles])
        finally:
            db.close()

        result = backtester.run(candles_by_tf, trainer.scalers)
        logger.info(
            f"[Retrain] New model backtest: WR={result.win_rate:.2%}, "
            f"PF={result.profit_factor:.2f}"
        )
    except Exception as e:
        logger.error(f"[Retrain] Backtest failed for {symbol}: {e}")
        _update_asset_status(symbol, "ACTIVE")
        return

    # Compare with existing
    db = SessionLocal()
    try:
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        old_wr = asset.backtest_win_rate or 0.0
        old_pf = asset.backtest_profit_factor or 0.0

        # Accept new model if it passes thresholds and is better or equal
        if result.passed and (
            result.win_rate >= old_wr or result.profit_factor >= old_pf
        ):
            model_path = trainer.save()
            asset.status = "ACTIVE"
            asset.model_path = model_path
            asset.last_trained = datetime.utcnow()
            asset.backtest_win_rate = result.win_rate
            asset.backtest_profit_factor = result.profit_factor
            db.commit()
            logger.info(f"[Retrain] {symbol} model updated successfully")
        else:
            # Rollback — keep existing model
            asset.status = "ACTIVE"
            db.commit()
            logger.info(
                f"[Retrain] {symbol} new model not better "
                f"(new WR={result.win_rate:.2%} vs old WR={old_wr:.2%}). "
                f"Keeping existing model."
            )
    finally:
        db.close()
