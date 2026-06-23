# Futures Bot — Implementation Plan

## Phase 1: Project Scaffolding & Database

### Task 1.1 — Project structure and dependencies
**Files**: `requirements.txt`, `src/__init__.py`, `src/config.py`, `.env.example`, `.gitignore`
**Do**:
- Create folder structure: `src/`, `src/data/`, `src/models/`, `src/features/`, `src/signals/`, `src/web/`, `src/notifications/`, `scripts/`, `tests/`
- Define all dependencies (pytorch, python-binance, fastapi, uvicorn, pandas, numpy, pandas-ta, sqlalchemy, mysql-connector-python, plyer, websockets, asyncio)
- Config loader from `.env` (API keys, MySQL creds, confidence threshold, ATR multipliers)
- `.gitignore` (`.env`, `*.pt`, `__pycache__`, `.venv`)
**Verify**: `pip install -r requirements.txt` succeeds. Config loads from `.env`.

### Task 1.2 — MySQL schema and database layer
**Files**: `src/data/database.py`, `src/data/models.py`, `scripts/init_db.sql`
**Do**:
- Tables: `candles` (asset, timeframe, timestamp, O, H, L, C, V), `signals` (asset, direction, entry, tp1, tp2, tp3, sl, confidence, status, created_at, resolved_at), `signal_results` (signal_id, tp1_hit, tp2_hit, tp3_hit, sl_hit, expired, resolved_price), `assets` (symbol, status, model_path, last_trained, backtest_wr, backtest_pf), `config` (key, value)
- SQLAlchemy ORM models
- Connection pool setup
**Verify**: `python scripts/init_db.sql` creates all tables. ORM can CRUD on each table.

---

## Phase 2: Data Pipeline

### Task 2.1 — Historical data fetcher
**Files**: `src/data/historical.py`
**Do**:
- Fetch 12 months of kline data from Binance REST API for a given asset and timeframe
- Handle pagination (Binance returns max 1000 candles per request)
- Store in MySQL `candles` table
- Progress callback for dashboard integration later
- Validate: no gaps, correct timestamps
**Verify**: Fetch 1 week of BTC/USDT 5m candles, confirm row count matches expected.

### Task 2.2 — WebSocket live stream manager
**Files**: `src/data/stream.py`
**Do**:
- Connect to Binance kline WebSocket for all active assets × all 6 timeframes
- Parse incoming candle data (open candle vs. closed candle)
- On candle close: store in MySQL, emit event to inference pipeline
- Auto-reconnect on disconnect with exponential backoff
- Backfill missed candles from REST API on reconnection
- Desktop notification on reconnect: "Reconnected after X minutes — gap backfilled"
**Verify**: Connect to BTC 5m stream, receive and store 3 closed candles correctly.

### Task 2.3 — Data integrity and gap detection
**Files**: `src/data/integrity.py`
**Do**:
- On startup: check for gaps in stored candles for each active asset/timeframe
- Backfill any gaps from REST API
- Log gaps found and filled
**Verify**: Delete 5 candles from DB, run integrity check, confirm they're backfilled.

---

## Phase 3: Feature Engineering

### Task 3.1 — Technical indicator calculator
**Files**: `src/features/indicators.py`
**Do**:
- Calculate from candle data: RSI(14), MACD(12/26/9), EMA 20/50/200, ATR(14), Bollinger Bands(20,2), volume delta
- Input: DataFrame of candles. Output: DataFrame with indicator columns appended.
- Handle edge cases: not enough data for longest EMA (200 periods)
**Verify**: Feed known candle data, assert indicator values match manual calculation for 3 data points.

### Task 3.2 — Funding rate and open interest fetcher
**Files**: `src/features/market_data.py`
**Do**:
- Fetch funding rate and open interest from Binance Futures API (requires read-only keys)
- Align timestamps with candle data
- Store in MySQL or merge on-the-fly
**Verify**: Fetch current funding rate for BTC, confirm it's a reasonable value.

### Task 3.3 — Feature pipeline (multi-timeframe)
**Files**: `src/features/pipeline.py`
**Do**:
- For a given asset: gather latest N candles across all 6 timeframes
- Run indicator calculator on each timeframe
- Normalize all features (min-max or z-score, per-feature)
- Stack into model input tensor: shape [6_timeframes × N_candles × N_features]
- Cache normalization params for inference consistency
**Verify**: Generate feature tensor for BTC, assert correct shape and no NaN values.

---

## Phase 4: Model Architecture & Training

### Task 4.1 — CNN-LSTM model definition
**Files**: `src/models/cnn_lstm.py`
**Do**:
- Define PyTorch model:
  - 6 parallel CNN branches (one per timeframe channel)
  - Each branch: Conv1D layers → BatchNorm → ReLU → pooling
  - Per-branch LSTM layer after CNN
  - Per-timeframe softmax head (output: long/short/neutral probabilities)
  - Merge layer: voting logic + confidence calculation
- Model outputs: direction (long/short/neutral), confidence (0-1), per-timeframe breakdown
**Verify**: Instantiate model, feed random tensor of correct shape, get output of correct shape.

### Task 4.2 — Training label generator
**Files**: `src/models/labeler.py`
**Do**:
- From historical candles: for each candle, compute ATR(14) at that point
- Look forward 50 candles
- Label LONG if price hit entry + 1×ATR before entry - 1.5×ATR
- Label SHORT if price hit entry - 1×ATR before entry + 1.5×ATR
- Label NO_TRADE otherwise
- Output: labeled dataset ready for training
**Verify**: Manually verify 5 labels against chart data for BTC.

### Task 4.3 — Training pipeline
**Files**: `src/models/trainer.py`, `scripts/train.py`
**Do**:
- Load candles from MySQL for asset (12 months)
- Generate labels (Task 4.2)
- Run feature pipeline (Task 3.3) on historical data
- Train/validation/test split (70/15/15, chronological — no shuffle)
- Training loop: Adam optimizer, cross-entropy loss, early stopping on val loss
- Dropout layers for regularization
- Save best model as `models/{asset}_latest.pt`
- Log training metrics (loss curves, accuracy per class)
**Verify**: Train on BTC 5m data (small subset), model loss decreases over epochs.

### Task 4.4 — Backtester
**Files**: `src/models/backtester.py`, `scripts/backtest.py`
**Do**:
- Replay historical candles through trained model (simulating live inference)
- Record signals generated: direction, entry, TP1/2/3, SL
- For each signal: check if TP1 or SL hit within 50 candles
- Track: win rate, average R:R, max drawdown, profit factor, per-TP hit rate
- Return pass/fail based on thresholds (>55% WR, >1.5 PF)
- Output detailed results for dashboard display
**Verify**: Run on test split, confirm metrics are calculated correctly.

---

## Phase 5: Live Inference & Signals

### Task 5.1 — Inference engine
**Files**: `src/signals/inference.py`
**Do**:
- On candle close event: run feature pipeline for asset → feed to model → get prediction
- Per-timeframe voting: check 2/3 primary (5m/10m/15m) agree
- Apply higher-timeframe multiplier (×1.2 agree, ×0.7 disagree, ×1.0 mixed)
- If confidence ≥ 70%: generate signal (calculate entry + ATR-based TP/SL levels)
- Hot-reload model from `.pt` file (for when retraining produces a new version)
**Verify**: Mock candle close event for BTC, confirm signal generation with correct levels.

### Task 5.2 — Signal manager and tracker
**Files**: `src/signals/manager.py`
**Do**:
- Store fired signal in MySQL with status "active"
- Watch live price stream for active signals
- On TP1 hit: mark tp1_hit, keep watching for TP2, TP3
- On SL hit: mark as loss, close signal
- On 50-candle expiry: mark as expired
- Maintain running stats: win rate over last 30 signals per asset
- Flag if win rate drops below 50%
**Verify**: Create a mock signal, simulate price hitting TP1, confirm status update.

### Task 5.3 — Desktop notification dispatcher
**Files**: `src/notifications/desktop.py`
**Do**:
- On new signal: fire Windows desktop notification
  - Title: "🟢 LONG BTC/USDT" or "🔴 SHORT BTC/USDT"
  - Body: "Entry: $X | TP1: $X | TP2: $X | TP3: $X | SL: $X | Conf: 78%"
- On reconnection: notify "Reconnected after X min — gap backfilled"
- On model degradation: notify "⚠️ BTC model win rate below 50%"
**Verify**: Fire a test notification, confirm it appears on screen.

---

## Phase 6: Web Dashboard

### Task 6.1 — FastAPI backend
**Files**: `src/web/app.py`, `src/web/routes.py`, `src/web/websocket.py`
**Do**:
- REST endpoints:
  - `GET /api/signals` — recent signals with status
  - `GET /api/signals/stats` — win rate, R:R, per-asset breakdown
  - `GET /api/assets` — list of active/inactive assets with model status
  - `POST /api/assets` — add new coin (triggers pipeline)
  - `DELETE /api/assets/{symbol}` — deactivate coin
  - `POST /api/assets/{symbol}/retrain` — trigger manual retrain
  - `GET /api/backtest/{symbol}` — backtest results
  - `GET /api/config` — current thresholds
  - `PUT /api/config` — update thresholds
- WebSocket endpoint: push new signals and status updates to connected clients in real-time
**Verify**: Hit each endpoint, confirm correct response shape.

### Task 6.2 — Frontend dashboard
**Files**: `src/web/static/index.html`, `src/web/static/app.js`, `src/web/static/style.css`
**Do**:
- Single-page layout:
  - **Signal feed**: live/recent signals with color coding (green long, red short), TP/SL levels, status (active/won/lost/expired)
  - **Stats panel**: per-asset win rate, overall stats, degradation warnings
  - **Assets panel**: list of coins, status (active/training/failed), add/remove buttons
  - **Config panel**: adjust confidence threshold, ATR multipliers
  - **Backtest tab**: per-asset backtest results with key metrics
- Real-time updates via WebSocket (no page refresh)
- Clean, minimal design — dark theme (trading standard)
**Verify**: Open in browser, confirm layout renders, WebSocket connects.

### Task 6.3 — New coin pipeline endpoint
**Files**: `src/web/tasks.py`
**Do**:
- When `POST /api/assets` is called:
  1. Validate symbol exists on Binance Futures
  2. Insert into `assets` table with status "fetching"
  3. Kick off background task: fetch 12mo → status "training" → train → status "backtesting" → backtest → status "active" or "failed"
  4. Push status updates to dashboard via WebSocket
**Verify**: Add "DOGE/USDT", watch status progress through pipeline on dashboard.

---

## Phase 7: Main Process & Integration

### Task 7.1 — Main entry point
**Files**: `src/main.py`
**Do**:
- On startup:
  1. Load config from `.env`
  2. Connect to MySQL, run integrity check
  3. Load models for all active assets
  4. Start WebSocket streams for all active assets × 6 timeframes
  5. Start FastAPI server (serves dashboard)
  6. Enter main event loop: candle close → features → inference → signal → notify
- Graceful shutdown on Ctrl+C (close streams, save state)
**Verify**: `python src/main.py` starts everything, dashboard accessible, streams running.

### Task 7.2 — Retraining scheduler
**Files**: `src/models/scheduler.py`
**Do**:
- Weekly timer (configurable day/time)
- For each active asset: retrain → backtest new model → compare to current → swap or rollback
- Log results, update dashboard
- Manual trigger via API also calls this
**Verify**: Trigger manual retrain for BTC, confirm new model saved and compared.

---

## Execution Order & Dependencies

```
1.1 → 1.2 → 2.1 → 2.2 → 2.3 (foundation)
         ↘
          3.1 → 3.2 → 3.3 (features, parallel with 2.x after 1.2)
                        ↓
                  4.1 → 4.2 → 4.3 → 4.4 (model)
                                     ↓
                              5.1 → 5.2 → 5.3 (signals)
                                           ↓
                                    6.1 → 6.2 → 6.3 (dashboard)
                                                  ↓
                                           7.1 → 7.2 (integration)
```

## Success Criteria

- [ ] System starts with one command, connects to Binance, receives live data
- [ ] Model trained on BTC/ETH/SOL with >55% win rate on test set
- [ ] Signals fire as desktop notifications with correct entry/TP/SL
- [ ] Dashboard shows live signals, stats, and allows adding new coins
- [ ] Retrain runs weekly without manual intervention
- [ ] New coin added via dashboard goes through full pipeline automatically
