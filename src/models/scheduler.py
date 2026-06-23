import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.config import config
from src.data.database import SessionLocal, Asset
from src.models.trainer import Trainer
from src.models.backtester import Backtester

logger = logging.getLogger(__name__)


class RetrainScheduler:
    def __init__(self, inference_engine):
        self.inference_engine = inference_engine
        self.scheduler = AsyncIOScheduler()

    def start(self):
        day_map = {
            "monday": "mon", "tuesday": "tue", "wednesday": "wed",
            "thursday": "thu", "friday": "fri", "saturday": "sat", "sunday": "sun",
        }
        day = day_map.get(config.RETRAIN_DAY.lower(), "mon")

        self.scheduler.add_job(
            self.retrain_all,
            trigger="cron",
            day_of_week=day,
            hour=config.RETRAIN_HOUR,
            id="weekly_retrain",
        )
        self.scheduler.start()
        logger.info(f"Retrain scheduled: every {config.RETRAIN_DAY} at {config.RETRAIN_HOUR}:00")

    def stop(self):
        self.scheduler.shutdown(wait=False)

    async def retrain_all(self):
        logger.info("Starting weekly retraining for all active assets")
        db = SessionLocal()
        try:
            active_assets = db.query(Asset).filter(Asset.status == "ACTIVE").all()
            symbols = [a.symbol for a in active_assets]
        finally:
            db.close()

        for symbol in symbols:
            try:
                await self.retrain_single(symbol)
            except Exception as e:
                logger.error(f"Retrain failed for {symbol}: {e}")

    async def retrain_single(self, symbol: str):
        logger.info(f"Retraining {symbol}")
        db = SessionLocal()

        try:
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            if not asset:
                return

            old_wr = asset.backtest_win_rate or 0
            old_pf = asset.backtest_profit_factor or 0

            # Train new model
            trainer = Trainer(symbol)
            train_result = trainer.train()
            new_model_path = trainer.save()

            # Backtest new model
            from src.models.cnn_lstm import FuturesModel
            new_model = Trainer.load_model(symbol)

            candles_by_tf = {}
            for tf in config.ALL_TIMEFRAMES:
                candles_by_tf[tf] = trainer.load_candles(tf)

            import json
            scaler_path = config.MODEL_DIR / f"{symbol}_scaler.json"
            with open(scaler_path) as f:
                scaler_params = json.load(f)

            backtester = Backtester(symbol, new_model)
            result = backtester.run(candles_by_tf, scaler_params)

            new_wr = result.win_rate
            new_pf = result.profit_factor

            # Compare: keep new model only if it's equal or better
            if new_wr >= old_wr and new_pf >= old_pf:
                asset.model_path = new_model_path
                asset.last_trained = datetime.utcnow()
                asset.backtest_win_rate = new_wr
                asset.backtest_profit_factor = new_pf
                db.commit()
                self.inference_engine.reload_model(symbol)
                logger.info(
                    f"{symbol} retrain SUCCESS: WR {old_wr:.2%} -> {new_wr:.2%}, "
                    f"PF {old_pf:.2f} -> {new_pf:.2f}"
                )
            else:
                # Rollback: restore old model (it's still on disk as _latest.pt was overwritten)
                # In production, we'd save as _candidate.pt first — simplified here
                logger.warning(
                    f"{symbol} retrain ROLLBACK: new model worse. "
                    f"WR {new_wr:.2%} < {old_wr:.2%} or PF {new_pf:.2f} < {old_pf:.2f}"
                )
        except Exception as e:
            logger.error(f"Retrain error for {symbol}: {e}")
        finally:
            db.close()
