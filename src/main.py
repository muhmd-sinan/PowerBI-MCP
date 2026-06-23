import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.config import config
from src.data.database import init_db, SessionLocal, Asset
from src.data.stream import StreamManager
from src.data.integrity import check_all_integrity
from src.signals.inference import InferenceEngine
from src.signals.manager import SignalManager
from src.notifications.desktop import notify_signal, notify_degradation
from src.models.scheduler import RetrainScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("futures_bot.log"),
    ],
)
logger = logging.getLogger(__name__)

inference_engine = InferenceEngine()
signal_manager = None
stream_manager = None
scheduler = None
ws_manager = None


async def on_signal_fired(signal_data: dict):
    notify_signal(signal_data)
    if ws_manager:
        await ws_manager.broadcast({"type": "new_signal", "signal": signal_data})


async def on_candle_close(symbol: str, timeframe: str, candle: dict):
    # Update signal tracking with new price data
    signal_manager.check_active_signals(
        symbol=symbol,
        current_price=candle["close"],
        current_high=candle["high"],
        current_low=candle["low"],
    )

    # Check for degradation
    if signal_manager.is_degraded(symbol):
        db = SessionLocal()
        try:
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            if asset:
                notify_degradation(symbol, asset.live_win_rate)
        finally:
            db.close()

    # Only run inference on primary timeframe closes
    if timeframe in config.PRIMARY_TIMEFRAMES:
        prediction = await inference_engine.predict(symbol)
        if prediction:
            await signal_manager.create_signal(prediction)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global signal_manager, stream_manager, scheduler, ws_manager

    logger.info("Starting Futures Bot...")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Run integrity check
    await check_all_integrity()
    logger.info("Data integrity check complete")

    # Load models for active assets
    db = SessionLocal()
    try:
        active_assets = db.query(Asset).filter(Asset.status == "ACTIVE").all()
        symbols = [a.symbol for a in active_assets]
    finally:
        db.close()

    for symbol in symbols:
        try:
            inference_engine.load_model(symbol)
        except FileNotFoundError:
            logger.warning(f"No model found for {symbol}, skipping")

    # Initialize signal manager
    signal_manager = SignalManager(on_signal=on_signal_fired)

    # Start WebSocket streams
    stream_manager = StreamManager(on_candle_close=on_candle_close)
    for symbol in symbols:
        await stream_manager.add_symbol(symbol)
    await stream_manager.start()
    logger.info(f"Streaming {len(symbols)} assets across {len(config.ALL_TIMEFRAMES)} timeframes")

    # Start retraining scheduler
    scheduler = RetrainScheduler(inference_engine)
    scheduler.start()
    logger.info("Retraining scheduler started")

    # Import and set ws_manager from web app
    from src.web.app import manager
    ws_manager = manager

    logger.info("Futures Bot fully operational")

    yield

    # Shutdown
    logger.info("Shutting down...")
    scheduler.stop()
    await stream_manager.stop()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    from src.web.app import create_fastapi_app
    app = create_fastapi_app(lifespan=lifespan)
    return app


def main():
    app = create_app()
    uvicorn.run(
        app,
        host=config.WEB_HOST,
        port=config.WEB_PORT,
        log_level="info",
    )


if __name__ == "__main__":
    main()
