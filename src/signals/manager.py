import logging
from datetime import datetime
from typing import Callable, Awaitable

from src.config import config
from src.data.database import SessionLocal, Signal, Asset

logger = logging.getLogger(__name__)


class SignalManager:
    COOLDOWN_SECONDS = 300  # 5 min dedup window per symbol+direction

    def __init__(self, on_signal: Callable[[dict], Awaitable[None]] = None):
        self.on_signal = on_signal
        self._last_signal: dict[str, datetime] = {}

    def _is_duplicate(self, symbol: str, direction: str) -> bool:
        key = f"{symbol}_{direction}"
        last = self._last_signal.get(key)
        if last and (datetime.utcnow() - last).total_seconds() < self.COOLDOWN_SECONDS:
            return True
        return False

    def _mark_fired(self, symbol: str, direction: str):
        self._last_signal[f"{symbol}_{direction}"] = datetime.utcnow()

    async def create_signal(self, signal_data: dict) -> Signal:
        symbol = signal_data["symbol"]
        direction = signal_data["direction"]

        if self._is_duplicate(symbol, direction):
            logger.debug(f"Skipping duplicate signal: {direction} {symbol}")
            return None

        db = SessionLocal()
        try:
            signal = Signal(
                symbol=symbol,
                direction=direction,
                entry_price=signal_data["entry_price"],
                tp1=signal_data["tp1"],
                tp2=signal_data["tp2"],
                tp3=signal_data["tp3"],
                sl=signal_data["sl"],
                confidence=signal_data["confidence"],
                status="ACTIVE",
            )
            db.add(signal)
            db.commit()
            db.refresh(signal)
            self._mark_fired(symbol, direction)
            logger.info(
                f"Signal #{signal.id}: {signal.direction} {signal.symbol} "
                f"@ {signal.entry_price:.2f} | Conf: {signal.confidence:.1%}"
            )

            if self.on_signal:
                await self.on_signal({
                    "id": signal.id,
                    "symbol": signal.symbol,
                    "direction": signal.direction,
                    "entry_price": float(signal.entry_price),
                    "tp1": float(signal.tp1),
                    "tp2": float(signal.tp2),
                    "tp3": float(signal.tp3),
                    "sl": float(signal.sl),
                    "confidence": float(signal.confidence),
                    "status": signal.status,
                    "created_at": signal.created_at.isoformat(),
                })

            return signal
        finally:
            db.close()

    def check_active_signals(self, symbol: str, current_price: float, current_high: float, current_low: float):
        db = SessionLocal()
        try:
            active_signals = (
                db.query(Signal)
                .filter(Signal.symbol == symbol, Signal.status == "ACTIVE")
                .all()
            )

            for signal in active_signals:
                signal.candles_elapsed += 1
                updated = False

                if signal.direction == "LONG":
                    if not signal.tp1_hit and current_high >= signal.tp1:
                        signal.tp1_hit = True
                        updated = True
                    if signal.tp1_hit and not signal.tp2_hit and current_high >= signal.tp2:
                        signal.tp2_hit = True
                        updated = True
                    if signal.tp2_hit and not signal.tp3_hit and current_high >= signal.tp3:
                        signal.tp3_hit = True
                        updated = True
                    if current_low <= signal.sl:
                        signal.status = "WON" if signal.tp1_hit else "LOST"
                        signal.resolved_at = datetime.utcnow()
                        updated = True
                else:  # SHORT
                    if not signal.tp1_hit and current_low <= signal.tp1:
                        signal.tp1_hit = True
                        updated = True
                    if signal.tp1_hit and not signal.tp2_hit and current_low <= signal.tp2:
                        signal.tp2_hit = True
                        updated = True
                    if signal.tp2_hit and not signal.tp3_hit and current_low <= signal.tp3:
                        signal.tp3_hit = True
                        updated = True
                    if current_high >= signal.sl:
                        signal.status = "WON" if signal.tp1_hit else "LOST"
                        signal.resolved_at = datetime.utcnow()
                        updated = True

                # Expiry check
                if signal.status == "ACTIVE" and signal.candles_elapsed >= config.FORWARD_WINDOW_CANDLES:
                    signal.status = "WON" if signal.tp1_hit else "EXPIRED"
                    signal.resolved_at = datetime.utcnow()
                    updated = True

                if updated:
                    logger.info(
                        f"Signal #{signal.id} {signal.symbol}: status={signal.status} "
                        f"TP1={signal.tp1_hit} TP2={signal.tp2_hit} TP3={signal.tp3_hit}"
                    )

            db.commit()
            self._update_asset_stats(symbol, db)
        finally:
            db.close()

    def _update_asset_stats(self, symbol: str, db):
        asset = db.query(Asset).filter(Asset.symbol == symbol).first()
        if not asset:
            return

        recent_signals = (
            db.query(Signal)
            .filter(
                Signal.symbol == symbol,
                Signal.status.in_(["WON", "LOST"]),
            )
            .order_by(Signal.resolved_at.desc())
            .limit(config.DEGRADATION_WINDOW)
            .all()
        )

        if not recent_signals:
            return

        wins = sum(1 for s in recent_signals if s.status == "WON")
        total = len(recent_signals)
        asset.live_win_rate = wins / total
        asset.total_signals = (
            db.query(Signal).filter(Signal.symbol == symbol).count()
        )
        db.commit()

    def get_stats(self, symbol: str = None) -> dict:
        db = SessionLocal()
        try:
            query = db.query(Signal).filter(Signal.status.in_(["WON", "LOST"]))
            if symbol:
                query = query.filter(Signal.symbol == symbol)

            resolved = query.all()
            if not resolved:
                return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}

            wins = sum(1 for s in resolved if s.status == "WON")
            losses = sum(1 for s in resolved if s.status == "LOST")
            total = wins + losses

            return {
                "total": total,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / total if total > 0 else 0,
                "tp1_hit_rate": sum(1 for s in resolved if s.tp1_hit) / total if total > 0 else 0,
                "tp2_hit_rate": sum(1 for s in resolved if s.tp2_hit) / total if total > 0 else 0,
                "tp3_hit_rate": sum(1 for s in resolved if s.tp3_hit) / total if total > 0 else 0,
            }
        finally:
            db.close()

    def is_degraded(self, symbol: str) -> bool:
        db = SessionLocal()
        try:
            asset = db.query(Asset).filter(Asset.symbol == symbol).first()
            if not asset or asset.live_win_rate is None:
                return False
            return asset.live_win_rate < config.DEGRADATION_THRESHOLD
        finally:
            db.close()
