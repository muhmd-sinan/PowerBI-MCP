"""Technical indicator calculations for candle data."""

import pandas as pd
import ta as ta_lib


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and append them as columns.

    Args:
        df: DataFrame with columns [open, high, low, close, volume].
            Must be sorted by time ascending.

    Returns:
        DataFrame with original columns plus all indicator columns.
        Rows without enough history will have NaN for those indicators.
    """
    df = df.copy()

    # RSI(14)
    df["rsi_14"] = ta_lib.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD(12, 26, 9)
    macd = ta_lib.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # EMAs
    df["ema_20"] = ta_lib.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema_50"] = ta_lib.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta_lib.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    # ATR(14)
    df["atr_14"] = ta_lib.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # Bollinger Bands(20, 2)
    bb = ta_lib.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_upper"] = bb.bollinger_hband()

    # Volume delta: positive if close > open (buying pressure), else negative
    df["volume_delta"] = df["volume"].where(
        df["close"] > df["open"], -df["volume"]
    )

    return df
