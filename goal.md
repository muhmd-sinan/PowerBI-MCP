# Futures Trading Signal Bot

## Overview

A cryptocurrency futures trading signal bot that watches live charts across multiple timeframes concurrently and signals high-confidence directional moves with specific entry, take-profit, and stop-loss levels. Delivered via desktop notifications with a web dashboard for monitoring and configuration.

## Architecture Decisions

### Signal System
- **Primary engine**: CNN-LSTM hybrid model (one model per asset)
- **LLM role**: None in the system. Opus used manually by the user as a sanity check.
- **Signal format**: Entry, TP1, TP2, TP3, SL
- **TP/SL calculation**: ATR-based (TP1 = 1×ATR, TP2 = 2×ATR, TP3 = 3×ATR, SL = 1.5×ATR)
- **Delivery**: Windows desktop notifications
- **Confidence threshold**: 70% to fire (no cooldown)

### Multi-Timeframe Strategy
- **Primary timeframes** (signal generation): 5m, 10m, 15m
- **Confluence timeframes** (confidence adjustment): 30m, 1h, 2h
- **Voting**: 2/3 primary timeframes must agree on direction
- **Multiplier**: Higher TFs agree → ×1.2 boost. Disagree → ×0.7 penalty. Mixed → ×1.0.

### Model Architecture
- **Type**: CNN-LSTM hybrid
- **Input**: 6 channels (one per timeframe)
- **CNN**: extracts local patterns per timeframe
- **LSTM**: captures temporal dependencies
- **Output**: per-timeframe softmax heads (long/short/neutral) → merged via voting + multiplier
- **Features**: Raw normalized OHLCV + RSI(14), MACD(12/26/9), EMA 20/50/200, ATR(14), Bollinger Bands(20,2), volume delta, funding rate, open interest

### Training
- **Data**: 12 months historical from Binance (minimum before model goes live)
- **Labels**: 50-candle forward-looking window. Long if TP1 hit before SL, Short if TP1 hit before SL (inverse), No-trade otherwise.
- **Hardware**: Local GPU (user has one)
- **Retraining**: Weekly on rolling 12-month window (auto-scheduled + manual trigger from dashboard)
- **Rollback**: If retrained model backtests worse than current → keep old model, warn on dashboard

### Assets
- **Starting set**: BTC/USDT, ETH/USDT, SOL/USDT
- **Extensible**: Add coins via web dashboard
- **New coin pipeline**: Validate pair → fetch 12mo data → train → backtest → auto-activate if passes thresholds, reject if not. Override available.
- **Activation thresholds**: >55% win rate AND >1.5 profit factor

### Data & Infrastructure
- **Live data**: Binance WebSocket kline streams (public + read-only API for funding/OI)
- **API keys**: Read-only + futures permissions, stored in `.env`
- **Database**: MySQL (candle data, signal history, config)
- **Model storage**: `.pt` files (PyTorch)
- **Startup**: Manual (`python main.py`), auto-reconnect with candle backfill on disconnect

### Tech Stack
- **Language**: Python (end-to-end)
- **ML**: PyTorch (CNN-LSTM)
- **Data**: pandas, numpy, ta-lib or pandas-ta for indicators
- **Streaming**: python-binance WebSocket
- **Web**: FastAPI + simple frontend
- **Notifications**: Windows desktop (plyer or win10toast)
- **Database**: MySQL (mysql-connector or SQLAlchemy)

### System Structure
- **One main process**: stream → features → inference → signal → notify + serves web dashboard
- **Separate scripts**: training pipeline, backtesting
- **Dashboard features**: live/recent signals with status, backtest results, model performance, add/remove coins, adjust thresholds, trigger retrain

### Signal Tracking
- Passive — system auto-evaluates its own signals
- Watches live price after signal fires for TP/SL hits
- 50-candle expiry if nothing hits
- Running stats: win rate, avg R:R, per-TP hit rate
- Red flag on dashboard if live win rate drops below 50% over last 30 signals

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Overfitting | Train/val/test splits, dropout, backtest gate before live |
| Market regime shift | Rolling window retrain, live win-rate monitor catches degradation |
| ATR compression in low-vol | May need minimum ATR floor to avoid micro-signals |
| Binance API changes | Use python-binance/ccxt which maintain compatibility |
