import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from src.config import config
from src.data.database import get_db, Signal, Asset, ConfigEntry
from src.web.tasks import add_coin_pipeline, retrain_asset

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Request/Response Models ---

class AddAssetRequest(BaseModel):
    symbol: str


class ConfigUpdateRequest(BaseModel):
    confidence_threshold: float | None = None
    atr_tp1_mult: float | None = None
    atr_tp2_mult: float | None = None
    atr_tp3_mult: float | None = None
    atr_sl_mult: float | None = None


# --- Signal Endpoints ---

@router.get("/api/signals")
def list_signals(
    symbol: str | None = None,
    status: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    query = db.query(Signal).order_by(Signal.created_at.desc())
    if symbol:
        query = query.filter(Signal.symbol == symbol)
    if status:
        query = query.filter(Signal.status == status)
    query = query.limit(limit)

    signals = query.all()
    return [
        {
            "id": s.id,
            "symbol": s.symbol,
            "direction": s.direction,
            "entry_price": s.entry_price,
            "tp1": s.tp1,
            "tp2": s.tp2,
            "tp3": s.tp3,
            "sl": s.sl,
            "confidence": s.confidence,
            "status": s.status,
            "tp1_hit": s.tp1_hit,
            "tp2_hit": s.tp2_hit,
            "tp3_hit": s.tp3_hit,
            "sl_hit": s.sl_hit,
            "candles_elapsed": s.candles_elapsed,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "resolved_at": s.resolved_at.isoformat() if s.resolved_at else None,
        }
        for s in signals
    ]


@router.get("/api/signals/stats")
def signal_stats(db: Session = Depends(get_db)):
    # Overall stats from resolved signals
    resolved = db.query(Signal).filter(Signal.status.in_(["WON", "LOST"])).all()
    total_resolved = len(resolved)
    wins = sum(1 for s in resolved if s.status == "WON")
    losses = sum(1 for s in resolved if s.status == "LOST")
    active_count = db.query(Signal).filter(Signal.status == "ACTIVE").count()
    total_signals = db.query(Signal).count()

    win_rate = wins / total_resolved if total_resolved > 0 else 0.0
    tp1_hits = sum(1 for s in resolved if s.tp1_hit)
    tp2_hits = sum(1 for s in resolved if s.tp2_hit)
    tp3_hits = sum(1 for s in resolved if s.tp3_hit)

    # Profit factor: sum of wins / sum of losses (using R multiples approximation)
    # TP1 = 1R, TP2 = 2R, TP3 = 3R, Loss = -1R
    gross_profit = tp1_hits * 1.0 + tp2_hits * 2.0 + tp3_hits * 3.0
    gross_loss = losses * 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Per-asset stats
    symbols = db.query(Signal.symbol).distinct().all()
    per_asset = {}
    for (sym,) in symbols:
        sym_resolved = [s for s in resolved if s.symbol == sym]
        sym_total = len(sym_resolved)
        sym_wins = sum(1 for s in sym_resolved if s.status == "WON")
        sym_tp1 = sum(1 for s in sym_resolved if s.tp1_hit)
        sym_tp2 = sum(1 for s in sym_resolved if s.tp2_hit)
        sym_tp3 = sum(1 for s in sym_resolved if s.tp3_hit)
        per_asset[sym] = {
            "total": sym_total,
            "wins": sym_wins,
            "win_rate": sym_wins / sym_total if sym_total > 0 else 0.0,
            "tp1_rate": sym_tp1 / sym_total if sym_total > 0 else 0.0,
            "tp2_rate": sym_tp2 / sym_total if sym_total > 0 else 0.0,
            "tp3_rate": sym_tp3 / sym_total if sym_total > 0 else 0.0,
        }

    return {
        "total_signals": total_signals,
        "active_signals": active_count,
        "total_resolved": total_resolved,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "tp1_hit_rate": round(tp1_hits / total_resolved, 4) if total_resolved > 0 else 0.0,
        "tp2_hit_rate": round(tp2_hits / total_resolved, 4) if total_resolved > 0 else 0.0,
        "tp3_hit_rate": round(tp3_hits / total_resolved, 4) if total_resolved > 0 else 0.0,
        "per_asset": per_asset,
    }


# --- Asset Endpoints ---

@router.get("/api/assets")
def list_assets(db: Session = Depends(get_db)):
    assets = db.query(Asset).order_by(Asset.created_at.desc()).all()
    return [
        {
            "id": a.id,
            "symbol": a.symbol,
            "status": a.status,
            "model_path": a.model_path,
            "last_trained": a.last_trained.isoformat() if a.last_trained else None,
            "backtest_win_rate": a.backtest_win_rate,
            "backtest_profit_factor": a.backtest_profit_factor,
            "live_win_rate": a.live_win_rate,
            "total_signals": a.total_signals,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in assets
    ]


@router.post("/api/assets")
def add_asset(
    body: AddAssetRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    symbol = body.symbol.upper().strip()
    if not symbol.endswith("USDT"):
        raise HTTPException(status_code=400, detail="Symbol must end with USDT")

    # Check if already exists
    existing = db.query(Asset).filter(Asset.symbol == symbol).first()
    if existing:
        if existing.status == "INACTIVE":
            existing.status = "FETCHING"
            db.commit()
            background_tasks.add_task(add_coin_pipeline, symbol)
            return {"message": f"Reactivating {symbol}", "symbol": symbol}
        raise HTTPException(status_code=409, detail=f"{symbol} already exists")

    # Create new asset
    asset = Asset(symbol=symbol, status="FETCHING")
    db.add(asset)
    db.commit()

    # Kick off background pipeline
    background_tasks.add_task(add_coin_pipeline, symbol)
    return {"message": f"Adding {symbol}", "symbol": symbol}


@router.delete("/api/assets/{symbol}")
def deactivate_asset(symbol: str, db: Session = Depends(get_db)):
    asset = db.query(Asset).filter(Asset.symbol == symbol.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    asset.status = "INACTIVE"
    db.commit()
    return {"message": f"{symbol} deactivated"}


@router.post("/api/assets/{symbol}/retrain")
def trigger_retrain(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    asset = db.query(Asset).filter(Asset.symbol == symbol.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    if asset.status not in ("ACTIVE", "FAILED"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retrain asset with status {asset.status}",
        )

    asset.status = "TRAINING"
    db.commit()
    background_tasks.add_task(retrain_asset, symbol.upper())
    return {"message": f"Retraining {symbol}"}


# --- Backtest Endpoint ---

@router.get("/api/backtest/{symbol}")
def get_backtest(symbol: str, db: Session = Depends(get_db)):
    asset = db.query(Asset).filter(Asset.symbol == symbol.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    return {
        "symbol": asset.symbol,
        "win_rate": asset.backtest_win_rate,
        "profit_factor": asset.backtest_profit_factor,
        "last_trained": asset.last_trained.isoformat() if asset.last_trained else None,
        "status": asset.status,
    }


# --- Config Endpoints ---

@router.get("/api/config")
def get_config(db: Session = Depends(get_db)):
    return {
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "atr_tp1_mult": config.ATR_TP1_MULT,
        "atr_tp2_mult": config.ATR_TP2_MULT,
        "atr_tp3_mult": config.ATR_TP3_MULT,
        "atr_sl_mult": config.ATR_SL_MULT,
    }


@router.put("/api/config")
def update_config(
    body: ConfigUpdateRequest,
    db: Session = Depends(get_db),
):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No config values provided")

    for key, value in updates.items():
        # Update in-memory config
        attr_name = key.upper()
        if hasattr(config, attr_name):
            setattr(config, attr_name, value)

        # Persist to DB
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == key).first()
        if entry:
            entry.value = str(value)
            entry.updated_at = datetime.utcnow()
        else:
            entry = ConfigEntry(key=key, value=str(value))
            db.add(entry)

    db.commit()
    return {"message": "Config updated", "updated": updates}
