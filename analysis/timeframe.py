"""Zaman dilimi (timeframe) tanımları ve yardımcıları."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeframeConfig:
    key: str
    label: str
    interval: str
    period: str
    lookback_default: int
    lookback_min: int
    lookback_max: int
    lookback_unit: str
    bars_1m: int
    bars_3m: int
    bars_6m: int
    min_bars: int
    candle_lookback: int
    pivot_window: int
    change_labels: tuple[str, str, str]
    note: str = ""


TIMEFRAMES: dict[str, TimeframeConfig] = {
    "1d": TimeframeConfig(
        key="1d",
        label="Günlük",
        interval="1d",
        period="1y",
        lookback_default=120,
        lookback_min=30,
        lookback_max=250,
        lookback_unit="mum (gün)",
        bars_1m=21,
        bars_3m=63,
        bars_6m=126,
        min_bars=30,
        candle_lookback=5,
        pivot_window=80,
        change_labels=("1A %", "3A %", "6A %"),
    ),
    "1wk": TimeframeConfig(
        key="1wk",
        label="Haftalık",
        interval="1wk",
        period="5y",
        lookback_default=80,
        lookback_min=30,
        lookback_max=200,
        lookback_unit="mum (hafta)",
        bars_1m=4,
        bars_3m=13,
        bars_6m=26,
        min_bars=30,
        candle_lookback=5,
        pivot_window=60,
        change_labels=("~1A %", "~3A %", "~6A %"),
        note="SMA200 için ~4 yıl haftalık veri kullanılır.",
    ),
    "1h": TimeframeConfig(
        key="1h",
        label="Saatlik",
        interval="1h",
        period="60d",
        lookback_default=120,
        lookback_min=30,
        lookback_max=300,
        lookback_unit="mum (saat)",
        bars_1m=35,
        bars_3m=105,
        bars_6m=210,
        min_bars=30,
        candle_lookback=8,
        pivot_window=80,
        change_labels=("~1A %", "~3A %", "~6A %"),
        note="BIST saatlik verisi Yahoo'da sınırlı olabilir; son ~60 gün.",
    ),
    "15m": TimeframeConfig(
        key="15m",
        label="15 Dakika",
        interval="15m",
        period="5d",
        lookback_default=100,
        lookback_min=20,
        lookback_max=200,
        lookback_unit="mum (15dk)",
        bars_1m=26,
        bars_3m=78,
        bars_6m=156,
        min_bars=20,
        candle_lookback=10,
        pivot_window=60,
        change_labels=("~1A %", "~3A %", "~6A %"),
        note="Kısa periyot; Yahoo en fazla ~5 günlük 15dk veri sunar.",
    ),
}

DEFAULT_TIMEFRAME = "1d"


def get_timeframe(key: str) -> TimeframeConfig:
    return TIMEFRAMES.get(key, TIMEFRAMES[DEFAULT_TIMEFRAME])


def timeframe_options() -> list[tuple[str, str]]:
    """(key, label) listesi — UI selectbox için."""
    return [(tf.key, tf.label) for tf in TIMEFRAMES.values()]
