"""Feature pipeline: prepares model input tensors from raw candle data."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sqlalchemy import desc

from src.config import config
from src.data.database import Candle, SessionLocal
from src.features.indicators import calculate_indicators
from src.features.market_data import get_market_features


# Indicator columns produced by calculate_indicators()
INDICATOR_COLS = [
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "ema_20", "ema_50", "ema_200",
    "atr_14",
    "bb_lower", "bb_mid", "bb_upper",
    "volume_delta",
]

# Base OHLCV columns used as features
OHLCV_COLS = ["open", "high", "low", "close", "volume"]

# Market microstructure features appended to every row
MARKET_COLS = ["funding_rate", "open_interest"]

# All feature columns in order
ALL_FEATURE_COLS = OHLCV_COLS + INDICATOR_COLS + MARKET_COLS


class FeaturePipeline:
    """Prepares normalized multi-timeframe feature tensors for the model.

    Output tensor shape: [num_timeframes, sequence_length, num_features]
    where num_timeframes = 6, sequence_length = 60 (default), num_features = 19.
    """

    def __init__(self, symbol: str, sequence_length: int = 60):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.timeframes = config.ALL_TIMEFRAMES  # 6 timeframes
        self.feature_cols = ALL_FEATURE_COLS

        # Normalization parameters: {feature_name: {"min": float, "max": float}}
        self.scaler_params: Optional[dict[str, dict[str, float]]] = None

    def get_num_features(self) -> int:
        """Return the number of features per timeframe."""
        return len(self.feature_cols)

    def _fetch_candles(self, timeframe: str, limit: int) -> pd.DataFrame:
        """Fetch the latest `limit` candles from the database for one timeframe."""
        db = SessionLocal()
        try:
            rows = (
                db.query(Candle)
                .filter(
                    Candle.symbol == self.symbol,
                    Candle.timeframe == timeframe,
                )
                .order_by(desc(Candle.timestamp))
                .limit(limit)
                .all()
            )
        finally:
            db.close()

        if not rows:
            return pd.DataFrame(columns=OHLCV_COLS)

        # Convert to DataFrame, oldest first
        data = [
            {
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
            for r in reversed(rows)
        ]
        return pd.DataFrame(data)

    def _build_features(
        self, df: pd.DataFrame, market: dict
    ) -> pd.DataFrame:
        """Run indicators and append market features to a candle DataFrame."""
        df = calculate_indicators(df)

        # Append market microstructure as constant columns (same for all rows)
        df["funding_rate"] = market.get("funding_rate", 0.0)
        df["open_interest"] = market.get("open_interest", 0.0)

        return df[self.feature_cols]

    def _fit_scaler(self, all_data: list[pd.DataFrame]) -> None:
        """Compute min/max per feature across all timeframes."""
        combined = pd.concat(all_data, ignore_index=True)
        self.scaler_params = {}
        for col in self.feature_cols:
            col_min = float(combined[col].min()) if not combined[col].isna().all() else 0.0
            col_max = float(combined[col].max()) if not combined[col].isna().all() else 1.0
            # Avoid division by zero
            if col_min == col_max:
                col_max = col_min + 1.0
            self.scaler_params[col] = {"min": col_min, "max": col_max}

    def _normalize(self, df: pd.DataFrame) -> np.ndarray:
        """Apply min-max normalization using stored scaler params.

        Returns numpy array of shape [rows, features].
        NaN values are replaced with 0 after normalization.
        """
        if self.scaler_params is None:
            raise RuntimeError(
                "Scaler not fitted. Call prepare_training() first or load_scaler()."
            )

        result = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            params = self.scaler_params[col]
            col_min = params["min"]
            col_max = params["max"]
            values = df[col].values.astype(np.float32)
            normalized = (values - col_min) / (col_max - col_min)
            # Clamp to [0, 1] and replace NaN with 0
            normalized = np.nan_to_num(normalized, nan=0.0)
            normalized = np.clip(normalized, 0.0, 1.0)
            result[:, i] = normalized

        return result

    def _to_tensor(self, arrays_by_tf: list[np.ndarray]) -> torch.Tensor:
        """Stack per-timeframe arrays into a single tensor.

        Each array is [sequence_length, num_features].
        Output: [num_timeframes, sequence_length, num_features].
        """
        stacked = np.stack(arrays_by_tf, axis=0)
        return torch.from_numpy(stacked)

    def _normalize_with_tf_scaler(self, df: pd.DataFrame, tf: str) -> np.ndarray:
        """Normalize using per-timeframe scaler format from Trainer.

        The Trainer saves scalers as {tf: {min: {col: val}, max: {col: val}}}.
        This method handles both that format and the flat {col: {min, max}} format.
        """
        if tf in self.scaler_params and isinstance(self.scaler_params[tf], dict):
            tf_params = self.scaler_params[tf]
            mins = tf_params["min"]
            maxs = tf_params["max"]
        else:
            return self._normalize(df)

        result = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            col_min = mins.get(col, 0.0)
            col_max = maxs.get(col, 1.0)
            if col_min == col_max:
                col_max = col_min + 1.0
            values = df[col].values.astype(np.float32)
            normalized = (values - col_min) / (col_max - col_min)
            normalized = np.nan_to_num(normalized, nan=0.0)
            normalized = np.clip(normalized, 0.0, 1.0)
            result[:, i] = normalized
        return result

    async def prepare_live(self) -> torch.Tensor:
        """Prepare a tensor from the latest live data for inference.

        Fetches candles from DB, computes indicators, fetches market data,
        normalizes, and returns tensor of shape [6, sequence_length, num_features].

        Requires scaler params to be loaded (call load_scaler() first).
        """
        if self.scaler_params is None:
            raise RuntimeError(
                "Scaler not loaded. Call load_scaler() before live inference."
            )

        # Fetch market features once (shared across timeframes)
        market = await get_market_features(self.symbol)

        arrays = []
        for tf in self.timeframes:
            # Fetch extra candles so indicators have warmup data
            # 200 extra for EMA200, plus sequence_length for the output window
            fetch_limit = self.sequence_length + 250
            df = self._fetch_candles(tf, fetch_limit)

            if df.empty or len(df) < self.sequence_length:
                # Not enough data — fill with zeros
                arr = np.zeros(
                    (self.sequence_length, self.get_num_features()),
                    dtype=np.float32,
                )
                arrays.append(arr)
                continue

            features_df = self._build_features(df, market)

            # Take only the last sequence_length rows (most recent)
            features_df = features_df.iloc[-self.sequence_length:]

            arr = self._normalize_with_tf_scaler(features_df, tf)
            arrays.append(arr)

        return self._to_tensor(arrays)

    def prepare_training(self, candles_by_timeframe: dict) -> torch.Tensor:
        """Prepare a tensor from provided historical candle data for training.

        Args:
            candles_by_timeframe: Dict mapping timeframe string to a DataFrame
                with columns [open, high, low, close, volume], sorted ascending.

        Returns:
            Tensor of shape [6, sequence_length, num_features].
            Also fits the scaler on this data.
        """
        # First pass: compute features for all timeframes
        feature_dfs = []
        market_placeholder = {"funding_rate": 0.0, "open_interest": 0.0}

        for tf in self.timeframes:
            df = candles_by_timeframe.get(tf, pd.DataFrame())
            if df.empty:
                # Create empty df with correct columns
                empty = pd.DataFrame(
                    np.zeros((self.sequence_length, len(self.feature_cols))),
                    columns=self.feature_cols,
                )
                feature_dfs.append(empty)
                continue

            features_df = self._build_features(df, market_placeholder)
            feature_dfs.append(features_df)

        # Fit scaler on all data combined
        self._fit_scaler(feature_dfs)

        # Second pass: normalize and take last sequence_length rows
        arrays = []
        for features_df in feature_dfs:
            tail = features_df.iloc[-self.sequence_length:]
            if len(tail) < self.sequence_length:
                # Pad with zeros at the front
                pad_rows = self.sequence_length - len(tail)
                padding = pd.DataFrame(
                    np.zeros((pad_rows, len(self.feature_cols))),
                    columns=self.feature_cols,
                )
                tail = pd.concat([padding, tail], ignore_index=True)

            arr = self._normalize(tail)
            arrays.append(arr)

        return self._to_tensor(arrays)

    def save_scaler(self, path: str) -> None:
        """Persist normalization parameters to a JSON file.

        Args:
            path: File path to write (e.g. "models/BTCUSDT_scaler.json").
        """
        if self.scaler_params is None:
            raise RuntimeError("No scaler params to save. Fit first.")

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.scaler_params, f, indent=2)

    def load_scaler(self, path: str) -> None:
        """Load normalization parameters from a JSON file.

        Args:
            path: File path to read (e.g. "models/BTCUSDT_scaler.json").
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(filepath, "r") as f:
            self.scaler_params = json.load(f)
