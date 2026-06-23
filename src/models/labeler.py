import numpy as np
import pandas as pd
from src.config import config


def generate_labels(df: pd.DataFrame, atr_series: pd.Series) -> pd.Series:
    """
    Label each candle as LONG (0), SHORT (1), or NO_TRADE (2).

    LONG: price hits entry + 1×ATR before entry - 1.5×ATR within 50 candles
    SHORT: price hits entry - 1×ATR before entry + 1.5×ATR within 50 candles
    NO_TRADE: neither condition met
    """
    forward_window = config.FORWARD_WINDOW_CANDLES
    labels = np.full(len(df), 2, dtype=np.int64)  # default NO_TRADE

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = atr_series.values

    for i in range(len(df) - forward_window):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue

        entry = closes[i]
        current_atr = atr[i]

        long_tp = entry + current_atr * config.ATR_TP1_MULT
        long_sl = entry - current_atr * config.ATR_SL_MULT
        short_tp = entry - current_atr * config.ATR_TP1_MULT
        short_sl = entry + current_atr * config.ATR_SL_MULT

        # Check forward window for LONG
        long_tp_hit = -1
        long_sl_hit = -1
        for j in range(i + 1, min(i + forward_window + 1, len(df))):
            if long_tp_hit == -1 and highs[j] >= long_tp:
                long_tp_hit = j
            if long_sl_hit == -1 and lows[j] <= long_sl:
                long_sl_hit = j
            if long_tp_hit != -1 and long_sl_hit != -1:
                break

        # Check forward window for SHORT
        short_tp_hit = -1
        short_sl_hit = -1
        for j in range(i + 1, min(i + forward_window + 1, len(df))):
            if short_tp_hit == -1 and lows[j] <= short_tp:
                short_tp_hit = j
            if short_sl_hit == -1 and highs[j] >= short_sl:
                short_sl_hit = j
            if short_tp_hit != -1 and short_sl_hit != -1:
                break

        # Determine label: TP must hit BEFORE SL
        long_valid = long_tp_hit != -1 and (long_sl_hit == -1 or long_tp_hit < long_sl_hit)
        short_valid = short_tp_hit != -1 and (short_sl_hit == -1 or short_tp_hit < short_sl_hit)

        if long_valid and short_valid:
            # Both valid — pick whichever TP hit sooner
            if long_tp_hit <= short_tp_hit:
                labels[i] = 0
            else:
                labels[i] = 1
        elif long_valid:
            labels[i] = 0
        elif short_valid:
            labels[i] = 1
        # else remains NO_TRADE (2)

    return pd.Series(labels, index=df.index)


def compute_label_stats(labels: pd.Series) -> dict:
    counts = labels.value_counts()
    total = len(labels)
    return {
        "total": total,
        "long": int(counts.get(0, 0)),
        "short": int(counts.get(1, 0)),
        "no_trade": int(counts.get(2, 0)),
        "long_pct": round(counts.get(0, 0) / total * 100, 1),
        "short_pct": round(counts.get(1, 0) / total * 100, 1),
        "no_trade_pct": round(counts.get(2, 0) / total * 100, 1),
    }
