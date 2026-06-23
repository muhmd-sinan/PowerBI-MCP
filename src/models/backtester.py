import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch

from src.config import config
from src.models.cnn_lstm import FuturesModel
from src.features.indicators import calculate_indicators

logger = logging.getLogger(__name__)


@dataclass
class BacktestSignal:
    index: int
    direction: str
    entry: float
    tp1: float
    tp2: float
    tp3: float
    sl: float
    confidence: float
    result: str = "PENDING"  # WON, LOST, EXPIRED
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    exit_price: float = 0.0
    pnl_r: float = 0.0  # profit/loss in R multiples


@dataclass
class BacktestResult:
    symbol: str
    total_signals: int = 0
    wins: int = 0
    losses: int = 0
    expired: int = 0
    tp1_hits: int = 0
    tp2_hits: int = 0
    tp3_hits: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_rr: float = 0.0
    max_drawdown: float = 0.0
    signals: List[BacktestSignal] = field(default_factory=list)
    passed: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "total_signals": self.total_signals,
            "wins": self.wins,
            "losses": self.losses,
            "expired": self.expired,
            "tp1_hits": self.tp1_hits,
            "tp2_hits": self.tp2_hits,
            "tp3_hits": self.tp3_hits,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_rr": round(self.avg_rr, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "passed": self.passed,
        }


class Backtester:
    def __init__(self, symbol: str, model: FuturesModel, device: str = None):
        self.symbol = symbol
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def run(
        self,
        candles_by_timeframe: dict,
        scaler_params: dict,
        sequence_length: int = 60,
    ) -> BacktestResult:
        logger.info(f"Running backtest for {self.symbol}")
        result = BacktestResult(symbol=self.symbol)

        # Prepare data — use primary 5m timeframe as the clock
        primary_tf = config.PRIMARY_TIMEFRAMES[0]
        primary_df = candles_by_timeframe[primary_tf].copy()

        # Calculate ATR on primary for TP/SL levels
        primary_with_indicators = calculate_indicators(primary_df.copy())
        atr_col = "atr_14" if "atr_14" in primary_with_indicators.columns else "ATRr_14"
        atr_values = primary_with_indicators[atr_col].values

        # Prepare features for all timeframes
        feature_dfs = {}
        feature_cols = None
        for tf in config.ALL_TIMEFRAMES:
            df = candles_by_timeframe[tf].copy()
            df = calculate_indicators(df)
            df["funding_rate"] = 0.0
            df["open_interest"] = 0.0
            df = df.dropna().reset_index(drop=True)
            if feature_cols is None:
                feature_cols = [c for c in df.columns if c != "timestamp"]
            feature_dfs[tf] = df

        num_features = len(feature_cols)
        min_len = min(len(df) for df in feature_dfs.values())
        usable_start = sequence_length
        usable_end = min_len - config.FORWARD_WINDOW_CANDLES

        # Normalize using provided scaler params
        for tf in config.ALL_TIMEFRAMES:
            df = feature_dfs[tf]
            if tf in scaler_params:
                mins = pd.Series(scaler_params[tf]["min"])
                maxs = pd.Series(scaler_params[tf]["max"])
                for col in feature_cols:
                    if col in mins.index:
                        range_val = maxs[col] - mins[col]
                        if range_val == 0:
                            range_val = 1
                        df[col] = (df[col] - mins[col]) / range_val
            feature_dfs[tf] = df

        # Walk through the data simulating live inference
        signals: List[BacktestSignal] = []

        for i in range(usable_start, usable_end):
            # Build input tensor for this moment
            X = np.zeros((1, len(config.ALL_TIMEFRAMES), sequence_length, num_features))
            for tf_idx, tf in enumerate(config.ALL_TIMEFRAMES):
                X[0, tf_idx] = feature_dfs[tf][feature_cols].values[i - sequence_length:i]

            X_tensor = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                output = self.model(X_tensor)

            direction = output["direction"].item()
            confidence = output["confidence"].item()
            agreement = output["agreement"].item()

            # Check thresholds
            if confidence < config.CONFIDENCE_THRESHOLD:
                continue
            if agreement < config.PRIMARY_VOTE_THRESHOLD:
                continue
            if direction == 2:  # neutral
                continue

            # Generate signal
            entry_price = primary_df["close"].iloc[i]
            current_atr = atr_values[i] if i < len(atr_values) and not np.isnan(atr_values[i]) else None
            if current_atr is None or current_atr == 0:
                continue

            if direction == 0:  # LONG
                signal = BacktestSignal(
                    index=i,
                    direction="LONG",
                    entry=entry_price,
                    tp1=entry_price + current_atr * config.ATR_TP1_MULT,
                    tp2=entry_price + current_atr * config.ATR_TP2_MULT,
                    tp3=entry_price + current_atr * config.ATR_TP3_MULT,
                    sl=entry_price - current_atr * config.ATR_SL_MULT,
                    confidence=confidence,
                )
            else:  # SHORT
                signal = BacktestSignal(
                    index=i,
                    direction="SHORT",
                    entry=entry_price,
                    tp1=entry_price - current_atr * config.ATR_TP1_MULT,
                    tp2=entry_price - current_atr * config.ATR_TP2_MULT,
                    tp3=entry_price - current_atr * config.ATR_TP3_MULT,
                    sl=entry_price + current_atr * config.ATR_SL_MULT,
                    confidence=confidence,
                )

            # Evaluate signal against future candles
            self._evaluate_signal(signal, primary_df, i)
            signals.append(signal)

        # Calculate metrics
        result.signals = signals
        result.total_signals = len(signals)

        if result.total_signals == 0:
            logger.warning(f"No signals generated during backtest for {self.symbol}")
            return result

        result.wins = sum(1 for s in signals if s.result == "WON")
        result.losses = sum(1 for s in signals if s.result == "LOST")
        result.expired = sum(1 for s in signals if s.result == "EXPIRED")
        result.tp1_hits = sum(1 for s in signals if s.tp1_hit)
        result.tp2_hits = sum(1 for s in signals if s.tp2_hit)
        result.tp3_hits = sum(1 for s in signals if s.tp3_hit)

        decided = result.wins + result.losses
        result.win_rate = result.wins / decided if decided > 0 else 0

        gross_profit = sum(s.pnl_r for s in signals if s.pnl_r > 0)
        gross_loss = abs(sum(s.pnl_r for s in signals if s.pnl_r < 0))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        pnls = [s.pnl_r for s in signals if s.result != "EXPIRED"]
        result.avg_rr = np.mean(pnls) if pnls else 0

        # Max drawdown in R
        cumulative = np.cumsum(pnls) if pnls else np.array([0])
        peak = np.maximum.accumulate(cumulative)
        drawdowns = peak - cumulative
        result.max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        # Pass/fail
        result.passed = (
            result.win_rate >= config.MIN_WIN_RATE
            and result.profit_factor >= config.MIN_PROFIT_FACTOR
        )

        logger.info(
            f"Backtest {self.symbol}: signals={result.total_signals}, "
            f"WR={result.win_rate:.2%}, PF={result.profit_factor:.2f}, "
            f"passed={result.passed}"
        )
        return result

    def _evaluate_signal(self, signal: BacktestSignal, df: pd.DataFrame, start_idx: int):
        window = config.FORWARD_WINDOW_CANDLES
        end_idx = min(start_idx + window + 1, len(df))

        for j in range(start_idx + 1, end_idx):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if signal.direction == "LONG":
                if not signal.tp1_hit and high >= signal.tp1:
                    signal.tp1_hit = True
                if signal.tp1_hit and not signal.tp2_hit and high >= signal.tp2:
                    signal.tp2_hit = True
                if signal.tp2_hit and not signal.tp3_hit and high >= signal.tp3:
                    signal.tp3_hit = True
                if low <= signal.sl:
                    signal.result = "LOST" if not signal.tp1_hit else "WON"
                    signal.exit_price = signal.sl
                    signal.pnl_r = -1.0 if not signal.tp1_hit else self._calc_pnl_r(signal)
                    return
            else:  # SHORT
                if not signal.tp1_hit and low <= signal.tp1:
                    signal.tp1_hit = True
                if signal.tp1_hit and not signal.tp2_hit and low <= signal.tp2:
                    signal.tp2_hit = True
                if signal.tp2_hit and not signal.tp3_hit and low <= signal.tp3:
                    signal.tp3_hit = True
                if high >= signal.sl:
                    signal.result = "LOST" if not signal.tp1_hit else "WON"
                    signal.exit_price = signal.sl
                    signal.pnl_r = -1.0 if not signal.tp1_hit else self._calc_pnl_r(signal)
                    return

        # Window expired
        if signal.tp1_hit:
            signal.result = "WON"
            signal.pnl_r = self._calc_pnl_r(signal)
        else:
            signal.result = "EXPIRED"
            signal.pnl_r = 0.0

    def _calc_pnl_r(self, signal: BacktestSignal) -> float:
        risk = abs(signal.entry - signal.sl)
        if risk == 0:
            return 0.0
        if signal.tp3_hit:
            reward = abs(signal.tp3 - signal.entry)
        elif signal.tp2_hit:
            reward = abs(signal.tp2 - signal.entry)
        else:
            reward = abs(signal.tp1 - signal.entry)
        return reward / risk
