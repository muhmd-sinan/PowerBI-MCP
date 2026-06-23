import logging
from datetime import datetime

import numpy as np
import torch

from src.config import config
from src.models.cnn_lstm import FuturesModel
from src.features.pipeline import FeaturePipeline
from src.features.indicators import calculate_indicators
from src.data.database import SessionLocal, Candle

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(self):
        self.models: dict[str, FuturesModel] = {}
        self.pipelines: dict[str, FeaturePipeline] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, symbol: str):
        from src.models.trainer import Trainer
        model = Trainer.load_model(symbol, self.device)
        self.models[symbol] = model
        self.pipelines[symbol] = FeaturePipeline(symbol)
        scaler_path = config.MODEL_DIR / f"{symbol}_scaler.json"
        if scaler_path.exists():
            self.pipelines[symbol].load_scaler(str(scaler_path))
        logger.info(f"Loaded model for {symbol} on {self.device}")

    def unload_model(self, symbol: str):
        self.models.pop(symbol, None)
        self.pipelines.pop(symbol, None)

    def reload_model(self, symbol: str):
        self.unload_model(symbol)
        self.load_model(symbol)
        logger.info(f"Hot-reloaded model for {symbol}")

    async def predict(self, symbol: str) -> dict | None:
        if symbol not in self.models:
            logger.warning(f"No model loaded for {symbol}")
            return None

        model = self.models[symbol]
        pipeline = self.pipelines[symbol]

        try:
            tensor = await pipeline.prepare_live()
        except Exception as e:
            logger.error(f"Feature preparation failed for {symbol}: {e}")
            return None

        # Add batch dimension: [1, 6, seq_len, features]
        X = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = model(X)

        direction = output["direction"].item()
        confidence = output["confidence"].item()
        agreement = output["agreement"].item()

        # Apply thresholds
        if confidence < config.CONFIDENCE_THRESHOLD:
            return None
        if agreement < config.PRIMARY_VOTE_THRESHOLD:
            return None
        if direction == 2:  # neutral
            return None

        # Get current ATR for TP/SL levels
        atr = self._get_current_atr(symbol)
        if atr is None or atr == 0:
            return None

        entry_price = self._get_latest_close(symbol)
        if entry_price is None:
            return None

        if direction == 0:  # LONG
            signal = {
                "symbol": symbol,
                "direction": "LONG",
                "entry_price": entry_price,
                "tp1": entry_price + atr * config.ATR_TP1_MULT,
                "tp2": entry_price + atr * config.ATR_TP2_MULT,
                "tp3": entry_price + atr * config.ATR_TP3_MULT,
                "sl": entry_price - atr * config.ATR_SL_MULT,
                "confidence": confidence,
                "agreement": agreement,
            }
        else:  # SHORT
            signal = {
                "symbol": symbol,
                "direction": "SHORT",
                "entry_price": entry_price,
                "tp1": entry_price - atr * config.ATR_TP1_MULT,
                "tp2": entry_price - atr * config.ATR_TP2_MULT,
                "tp3": entry_price - atr * config.ATR_TP3_MULT,
                "sl": entry_price + atr * config.ATR_SL_MULT,
                "confidence": confidence,
                "agreement": agreement,
            }

        return signal

    def _get_current_atr(self, symbol: str) -> float | None:
        import pandas as pd
        db = SessionLocal()
        try:
            candles = (
                db.query(Candle)
                .filter(Candle.symbol == symbol, Candle.timeframe == "5m")
                .order_by(Candle.timestamp.desc())
                .limit(20)
                .all()
            )
            if len(candles) < 15:
                return None
            candles.reverse()
            df = pd.DataFrame([{
                "open": c.open, "high": c.high, "low": c.low,
                "close": c.close, "volume": c.volume,
            } for c in candles])
            df = calculate_indicators(df)
            atr_col = "atr_14" if "atr_14" in df.columns else "ATRr_14"
            atr_val = df[atr_col].iloc[-1]
            return float(atr_val) if not np.isnan(atr_val) else None
        finally:
            db.close()

    def _get_latest_close(self, symbol: str) -> float | None:
        db = SessionLocal()
        try:
            candle = (
                db.query(Candle)
                .filter(Candle.symbol == symbol, Candle.timeframe == "5m")
                .order_by(Candle.timestamp.desc())
                .first()
            )
            return candle.close if candle else None
        finally:
            db.close()
