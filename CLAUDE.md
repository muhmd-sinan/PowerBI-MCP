# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cryptocurrency futures trading signal bot. CNN-LSTM model per asset watches 6 timeframes (5m/10m/15m primary, 30m/1h/2h confluence), fires desktop notifications with Entry/TP1/TP2/TP3/SL levels when confidence ≥ 70%.

## Commands

```bash
# Setup
pip install -r requirements.txt
python -c "from src.data.database import init_db; init_db()"

# Run the bot (streams + inference + dashboard)
python src/main.py

# Train a model (--fetch to download data first)
python scripts/train.py BTCUSDT --fetch

# Backtest a trained model
python scripts/backtest.py BTCUSDT

# Dashboard accessible at http://localhost:8000
```

## Architecture

```
src/main.py            — Entry point. Starts streams, inference loop, scheduler, and dashboard.
src/config.py          — All settings from .env (thresholds, timeframes, DB, API keys)
src/data/              — Binance WebSocket streams, REST historical fetcher, integrity checker
src/features/          — Technical indicators (pandas_ta), funding/OI fetcher, feature pipeline → tensor
src/models/            — CNN-LSTM model, label generator, trainer, backtester, retrain scheduler
src/signals/           — Inference engine (model → prediction), signal manager (tracking + stats)
src/notifications/     — Windows desktop notifications via plyer
src/web/               — FastAPI dashboard + WebSocket push + static SPA
scripts/               — Standalone train.py and backtest.py
```

## Key Design Decisions

- **One model per asset** — BTC/ETH/SOL default. New coins added via dashboard auto-pipeline.
- **ATR-based levels** — TP1=1×ATR, TP2=2×ATR, TP3=3×ATR, SL=1.5×ATR
- **Multi-TF voting** — 2/3 primary must agree. Confluence multiplier: ×1.2 boost, ×0.7 penalty.
- **Passive signal tracking** — System auto-evaluates signals (TP/SL hit detection, 50-candle expiry)
- **Auto-rollback** — Weekly retrain only deploys if backtest ≥ existing metrics
- **10m timeframe** — Not native to Binance; aggregated from 5m candles

## Database

MySQL (`futures_bot`). Tables: candles, signals, assets, config. ORM via SQLAlchemy in `src/data/database.py`.

## Conventions

- Never commit `.env`, `*.pt` model files, or `credentials.json`
- Config changes go through `.env` or the dashboard API
- All async code uses Python asyncio
- Model weights in `models/{SYMBOL}_latest.pt` with `{SYMBOL}_scaler.json`
