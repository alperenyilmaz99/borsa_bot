"""Teknik analiz göstergeleri — ta kütüphanesi ile hesaplanır."""

from __future__ import annotations

import math

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

from analysis.timeframe import DEFAULT_TIMEFRAME, get_timeframe


def calculate_indicators(
    df: pd.DataFrame,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> dict:
    """Tüm göstergeleri hesaplar, son güncel değerleri dict olarak döner."""
    tf = get_timeframe(timeframe_key)
    if df is None or df.empty or len(df) < tf.min_bars:
        return {}

    enriched = _enrich_dataframe(df)
    if enriched.empty:
        return {}

    row = enriched.iloc[-1]
    prev = enriched.iloc[-2] if len(enriched) > 1 else row
    close = _safe_float(row.get("Close"))

    sma_20 = _safe_float(row.get("SMA_20"))
    sma_50 = _safe_float(row.get("SMA_50"))
    sma_200 = _safe_float(row.get("SMA_200"))
    volume = _safe_float(row.get("Volume"))
    volume_sma_20 = _safe_float(row.get("Volume_SMA_20"))
    volume_ratio = _safe_float(row.get("Volume_Ratio"))

    bb_upper = _safe_float(row.get("BB_Upper"))
    bb_middle = _safe_float(row.get("BB_Middle"))
    bb_lower = _safe_float(row.get("BB_Lower"))

    return {
        "close": close,
        "rsi_14": _safe_float(row.get("RSI_14")),
        "macd": _safe_float(row.get("MACD")),
        "macd_signal_line": _safe_float(row.get("MACD_Signal")),
        "macd_hist": _safe_float(row.get("MACD_Hist")),
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "volume": volume,
        "volume_sma_20": volume_sma_20,
        "volume_ratio": volume_ratio,
        "price_change_1m": _price_change_pct(enriched, tf.bars_1m),
        "price_change_3m": _price_change_pct(enriched, tf.bars_3m),
        "price_change_6m": _price_change_pct(enriched, tf.bars_6m),
        "above_sma_20": _above_sma(close, sma_20),
        "above_sma_50": _above_sma(close, sma_50),
        "above_sma_200": _above_sma(close, sma_200),
        "golden_death_cross": _golden_death_cross(row, prev),
        "sma_trend": _sma_trend_label(close, sma_20, sma_50),
        "macd_signal": _macd_signal_label(row, prev),
        "rsi_signal": _rsi_signal_label(_safe_float(row.get("RSI_14"))),
        "volume_signal": _volume_signal_label(volume_ratio),
        "adx_strength": _trend_strength_label(enriched, tf.bars_3m),
        "bb_signal": _bb_signal_label(close, bb_upper, bb_middle, bb_lower),
    }


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'ine gösterge sütunları ekler (grafik için)."""
    if df is None or df.empty:
        return pd.DataFrame()
    return _enrich_dataframe(df)


def get_latest_signals(df: pd.DataFrame) -> dict:
    """scorer.py uyumluluğu için sinyal dict'i döner."""
    return calculate_indicators(df)


def _enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = data["Close"]

    data["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    data["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    data["SMA_200"] = SMAIndicator(close=close, window=200).sma_indicator()

    macd = MACD(close=close)
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()
    data["MACD_Hist"] = macd.macd_diff()

    data["RSI_14"] = RSIIndicator(close=close, window=14).rsi()

    data["Volume_SMA_20"] = SMAIndicator(close=data["Volume"], window=20).sma_indicator()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA_20"]

    bb = BollingerBands(close=close, window=20, window_dev=2)
    data["BB_Upper"] = bb.bollinger_hband()
    data["BB_Middle"] = bb.bollinger_mavg()
    data["BB_Lower"] = bb.bollinger_lband()

    return data


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _above_sma(close: float | None, sma: float | None) -> bool | None:
    if close is None or sma is None:
        return None
    return close > sma


def _price_change_pct(df: pd.DataFrame, days: int) -> float | None:
    if len(df) <= days:
        return None
    old_close = _safe_float(df["Close"].iloc[-days - 1])
    new_close = _safe_float(df["Close"].iloc[-1])
    if old_close is None or new_close is None or old_close == 0:
        return None
    return round((new_close - old_close) / old_close * 100, 2)


def _golden_death_cross(row: pd.Series, prev: pd.Series) -> str:
    sma_50 = _safe_float(row.get("SMA_50"))
    sma_200 = _safe_float(row.get("SMA_200"))
    prev_50 = _safe_float(prev.get("SMA_50"))
    prev_200 = _safe_float(prev.get("SMA_200"))

    if any(v is None for v in (sma_50, sma_200, prev_50, prev_200)):
        return "neutral"

    if prev_50 <= prev_200 and sma_50 > sma_200:
        return "golden_cross"
    if prev_50 >= prev_200 and sma_50 < sma_200:
        return "death_cross"
    if sma_50 > sma_200:
        return "golden"
    if sma_50 < sma_200:
        return "death"
    return "neutral"


def _sma_trend_label(
    close: float | None, sma_20: float | None, sma_50: float | None
) -> str:
    if close is None or sma_20 is None or sma_50 is None:
        return "neutral"
    if close > sma_20 > sma_50:
        return "bullish"
    if close < sma_20 < sma_50:
        return "bearish"
    return "neutral"


def _macd_signal_label(row: pd.Series, prev: pd.Series) -> str:
    macd = _safe_float(row.get("MACD"))
    signal = _safe_float(row.get("MACD_Signal"))
    prev_macd = _safe_float(prev.get("MACD"))
    prev_signal = _safe_float(prev.get("MACD_Signal"))

    if any(v is None for v in (macd, signal, prev_macd, prev_signal)):
        return "neutral"
    if macd > signal and prev_macd <= prev_signal:
        return "bullish_cross"
    if macd < signal and prev_macd >= prev_signal:
        return "bearish_cross"
    if macd > signal:
        return "bullish"
    if macd < signal:
        return "bearish"
    return "neutral"


def _rsi_signal_label(rsi: float | None) -> str:
    if rsi is None:
        return "neutral"
    if rsi < 30:
        return "oversold"
    if rsi > 70:
        return "overbought"
    if rsi >= 50:
        return "bullish"
    return "bearish"


def _volume_signal_label(volume_ratio: float | None) -> str:
    if volume_ratio is None:
        return "normal"
    if volume_ratio > 1.5:
        return "high_volume"
    if volume_ratio < 0.5:
        return "low_volume"
    return "normal"


def _bb_signal_label(
    close: float | None,
    upper: float | None,
    middle: float | None,
    lower: float | None,
) -> str:
    if any(v is None for v in (close, upper, middle, lower)):
        return "neutral"
    if close >= upper:
        return "overbought"
    if close <= lower:
        return "oversold"
    if close > middle:
        return "bullish"
    if close < middle:
        return "bearish"
    return "neutral"


def _trend_strength_label(df: pd.DataFrame, bars_3m: int) -> str:
    change_3m = _price_change_pct(df, bars_3m)
    if change_3m is None:
        return "weak_trend"
    if abs(change_3m) >= 15:
        return "strong_trend"
    if abs(change_3m) >= 5:
        return "moderate_trend"
    return "weak_trend"
