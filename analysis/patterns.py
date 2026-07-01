"""Formasyon tespiti — mum çubukları (TA-Lib) ve geometrik (deneysel)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from analysis.timeframe import DEFAULT_TIMEFRAME, get_timeframe
from analysis.indicators import add_indicators

logger = logging.getLogger(__name__)

try:
    import talib

    HAS_TALIB = True
except ImportError:
    talib = None  # type: ignore
    HAS_TALIB = False
    logger.warning("TA-Lib kurulu değil — mum formasyonları atlanacak.")

MIN_BARS = 30
PRICE_TOLERANCE = 0.025

CANDLE_FUNCTIONS: dict[str, str] = {
    "CDLDOJI": "Doji",
    "CDLHAMMER": "Hammer (Çekiç)",
    "CDLINVERTEDHAMMER": "Inverted Hammer (Ters Çekiç)",
    "CDLENGULFINGBULL": "Bullish Engulfing (Yutan Boğa)",
    "CDLENGULFINGBEAR": "Bearish Engulfing (Yutan Ayı)",
    "CDLMORNINGSTAR": "Morning Star (Sabah Yıldızı)",
    "CDLEVENINGSTAR": "Evening Star (Akşam Yıldızı)",
    "CDLHANGINGMAN": "Hanging Man (Asılmış Adam)",
    "CDLSHOOTINGSTAR": "Shooting Star (Kayan Yıldız)",
    "CDLPIERCING": "Piercing Line (Delici Çizgi)",
    "CDLDARKCLOUDCOVER": "Dark Cloud Cover (Kara Bulut)",
    "CDL3WHITESOLDIERS": "Three White Soldiers (Üç Beyaz Asker)",
    "CDL3BLACKCROWS": "Three Black Crows (Üç Kara Karga)",
    "CDLMARUBOZU": "Marubozu",
    "CDLSPINNINGTOP": "Spinning Top (Topaç)",
}

PATTERN_CATEGORIES = {
    "mum": "Mum Formasyonu",
    "geometrik": "Grafik Formasyonu",
    "trend": "Trend Sinyali",
}


PIVOT_ORDER = 5


def detect_patterns(
    df: pd.DataFrame,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> list[dict[str, Any]]:
    """Bulunan formasyonları isim, güven, kategori ve grafik bölgeleriyle döner."""
    tf = get_timeframe(timeframe_key)
    if df is None or df.empty or len(df) < tf.min_bars:
        return []

    patterns: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in _detect_candlestick_patterns(df, tf.candle_lookback):
        key = item["name"]
        if key not in seen:
            seen.add(key)
            patterns.append(item)

    for item in _detect_geometric_patterns(df, tf.pivot_window):
        key = item["name"]
        if key not in seen:
            seen.add(key)
            patterns.append(item)

    return patterns


def patterns_summary_table(patterns: list[dict[str, Any]]) -> pd.DataFrame:
    """UI tablosu için formasyon özet DataFrame."""
    if not patterns:
        return pd.DataFrame(columns=["Formasyon", "Kategori", "Güven", "Açıklama"])
    rows = []
    for p in patterns:
        conf = p.get("confidence", "deneysel")
        conf_label = "Kesin (TA-Lib / matematik)" if conf == "kesin" else "Deneysel (heuristik)"
        rows.append({
            "Formasyon": p.get("name", "-"),
            "Kategori": PATTERN_CATEGORIES.get(p.get("category", ""), p.get("category", "-")),
            "Güven": conf_label,
            "Açıklama": p.get("description", "-"),
        })
    return pd.DataFrame(rows)


def format_patterns_for_display(patterns: list[dict[str, Any]]) -> str:
    if not patterns:
        return "Tespit edilen formasyon yok"
    lines = []
    for p in patterns:
        conf = p.get("confidence", "deneysel")
        lines.append(f"- {p['name']} [{conf}]")
    return "\n".join(lines)


def _abs_idx(df_len: int, window_start: int, local_idx: int) -> int:
    return window_start + local_idx


def _detect_candlestick_patterns(
    df: pd.DataFrame,
    candle_lookback: int,
) -> list[dict[str, Any]]:
    if not HAS_TALIB:
        return []

    open_ = df["Open"].astype(float).values
    high = df["High"].astype(float).values
    low = df["Low"].astype(float).values
    close = df["Close"].astype(float).values

    found: list[dict[str, Any]] = []
    start = max(0, len(df) - candle_lookback)

    for func_name, label in CANDLE_FUNCTIONS.items():
        func = getattr(talib, func_name, None)
        if func is None:
            continue
        try:
            result = func(open_, high, low, close)
        except Exception as exc:
            logger.debug("%s hesaplanamadı: %s", func_name, exc)
            continue

        for i in range(start, len(result)):
            if result[i] == 0:
                continue
            direction = "Boğa" if result[i] > 0 else "Ayı"
            bar_offset = len(df) - 1 - i
            suffix = f" (son-{bar_offset} mum)" if bar_offset > 0 else " (son mum)"
            name = f"{label} [{direction}]{suffix}"
            found.append({
                "name": name,
                "confidence": "kesin",
                "category": "mum",
                "description": f"TA-Lib {label} formasyonu tespit edildi.",
                "regions": [{
                    "kind": "marker",
                    "idx": i,
                    "price": float(high[i]),
                    "style": "marker",
                }],
            })
            break

    return found


def _detect_geometric_patterns(
    df: pd.DataFrame,
    pivot_window: int,
) -> list[dict[str, Any]]:
    enriched = add_indicators(df)
    if enriched.empty:
        return []

    found: list[dict[str, Any]] = []
    high = enriched["High"].astype(float).values
    low = enriched["Low"].astype(float).values
    close = enriched["Close"].astype(float).values

    window = min(pivot_window, len(enriched))
    window_start = len(enriched) - window
    h = high[-window:]
    l = low[-window:]
    c = close[-window:]

    peak_idx = argrelextrema(h, np.greater, order=PIVOT_ORDER)[0]
    trough_idx = argrelextrema(l, np.less, order=PIVOT_ORDER)[0]

    peaks = [(i, h[i]) for i in peak_idx]
    troughs = [(i, l[i]) for i in trough_idx]

    found.extend(_detect_sma_cross(enriched))
    found.extend(_detect_triangles(peaks, troughs, window_start))
    found.extend(_detect_double_top_bottom(peaks, troughs, window_start))
    found.extend(_detect_head_shoulders(peaks, troughs, window_start, h))
    found.extend(_detect_cup_handle(c, troughs, window_start))
    found.extend(_detect_flag(c, window_start))

    return found


def _detect_sma_cross(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    if len(enriched) < 2 or "SMA_50" not in enriched or "SMA_200" not in enriched:
        return []

    row = enriched.iloc[-1]
    prev = enriched.iloc[-2]
    s50, s200 = row["SMA_50"], row["SMA_200"]
    p50, p200 = prev["SMA_50"], prev["SMA_200"]

    if any(pd.isna(v) for v in (s50, s200, p50, p200)):
        return []

    cross_idx = len(enriched) - 1
    price = float(row["Close"])

    if p50 <= p200 and s50 > s200:
        return [_pattern(
            "Golden Cross (SMA50 > SMA200)", "kesin", "trend",
            "SMA50, SMA200'ü yukarı kesti.",
            [{"kind": "marker", "idx": cross_idx, "price": price, "style": "trend"}],
        )]
    if p50 >= p200 and s50 < s200:
        return [_pattern(
            "Death Cross (SMA50 < SMA200)", "kesin", "trend",
            "SMA50, SMA200'ü aşağı kesti.",
            [{"kind": "marker", "idx": cross_idx, "price": price, "style": "trend"}],
        )]
    if s50 > s200:
        return [_pattern(
            "Golden Cross aktif (SMA50 > SMA200)", "kesin", "trend",
            "Uzun vadeli yükseliş yapısı aktif.",
            [],
        )]
    if s50 < s200:
        return [_pattern(
            "Death Cross aktif (SMA50 < SMA200)", "kesin", "trend",
            "Uzun vadeli düşüş yapısı aktif.",
            [],
        )]
    return []


def _detect_triangles(
    peaks: list[tuple[int, float]],
    troughs: list[tuple[int, float]],
    window_start: int,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if len(peaks) < 2 or len(troughs) < 2:
        return found

    recent_peaks = peaks[-3:]
    recent_troughs = troughs[-3:]

    peak_prices = [p[1] for p in recent_peaks]
    peak_std = np.std(peak_prices) / np.mean(peak_prices) if np.mean(peak_prices) else 1.0

    trough_x = [t[0] for t in recent_troughs]
    trough_y = [t[1] for t in recent_troughs]
    trough_slope = _slope(trough_x, trough_y)

    peak_x = [p[0] for p in recent_peaks]
    peak_y = [p[1] for p in recent_peaks]
    peak_slope = _slope(peak_x, peak_y)

    if peak_std < PRICE_TOLERANCE and trough_slope > 0:
        regions = [
            {
                "kind": "line",
                "points": [
                    (_abs_idx(0, window_start, peak_x[0]), peak_y[0]),
                    (_abs_idx(0, window_start, peak_x[-1]), peak_y[-1]),
                ],
                "label": "Direnç",
                "style": "triangle",
            },
            {
                "kind": "line",
                "points": [
                    (_abs_idx(0, window_start, trough_x[0]), trough_y[0]),
                    (_abs_idx(0, window_start, trough_x[-1]), trough_y[-1]),
                ],
                "label": "Yükselen destek",
                "style": "triangle",
            },
        ]
        found.append(_pattern(
            "Yükselen üçgen", "deneysel", "geometrik",
            "Yatay direnç + yükselen dipler; kırılım beklentisi izlenir.",
            regions,
        ))

    trough_std = np.std(trough_y) / np.mean(trough_y) if np.mean(trough_y) else 1.0
    if trough_std < PRICE_TOLERANCE and peak_slope < 0:
        regions = [
            {
                "kind": "line",
                "points": [
                    (_abs_idx(0, window_start, trough_x[0]), trough_y[0]),
                    (_abs_idx(0, window_start, trough_x[-1]), trough_y[-1]),
                ],
                "label": "Destek",
                "style": "triangle",
            },
            {
                "kind": "line",
                "points": [
                    (_abs_idx(0, window_start, peak_x[0]), peak_y[0]),
                    (_abs_idx(0, window_start, peak_x[-1]), peak_y[-1]),
                ],
                "label": "Alçalan direnç",
                "style": "triangle",
            },
        ]
        found.append(_pattern(
            "Alçalan üçgen", "deneysel", "geometrik",
            "Yatay destek + alçalan tepeler.",
            regions,
        ))

    return found


def _detect_double_top_bottom(
    peaks: list[tuple[int, float]],
    troughs: list[tuple[int, float]],
    window_start: int,
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        avg = (p1[1] + p2[1]) / 2
        if avg > 0 and abs(p1[1] - p2[1]) / avg < PRICE_TOLERANCE and p2[0] - p1[0] >= PIVOT_ORDER:
            found.append(_pattern(
                "Çift tepe", "deneysel", "geometrik",
                "İki benzer tepe seviyesi; direnç bölgesi.",
                [
                    {"kind": "marker", "idx": _abs_idx(0, window_start, p1[0]), "price": p1[1], "style": "double"},
                    {"kind": "marker", "idx": _abs_idx(0, window_start, p2[0]), "price": p2[1], "style": "double"},
                    {"kind": "hline", "price": avg, "label": "Çift tepe", "style": "double"},
                ],
            ))

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        avg = (t1[1] + t2[1]) / 2
        if avg > 0 and abs(t1[1] - t2[1]) / avg < PRICE_TOLERANCE and t2[0] - t1[0] >= PIVOT_ORDER:
            found.append(_pattern(
                "Çift dip", "deneysel", "geometrik",
                "İki benzer dip seviyesi; destek bölgesi.",
                [
                    {"kind": "marker", "idx": _abs_idx(0, window_start, t1[0]), "price": t1[1], "style": "double"},
                    {"kind": "marker", "idx": _abs_idx(0, window_start, t2[0]), "price": t2[1], "style": "double"},
                    {"kind": "hline", "price": avg, "label": "Çift dip", "style": "double"},
                ],
            ))

    return found


def _detect_head_shoulders(
    peaks: list[tuple[int, float]],
    troughs: list[tuple[int, float]],
    window_start: int,
    high: np.ndarray,
) -> list[dict[str, Any]]:
    """Basit omuz-baş-omuz heuristiği."""
    if len(peaks) < 3:
        return []

    left, head, right = peaks[-3], peaks[-2], peaks[-1]
    if not (left[1] < head[1] and right[1] < head[1]):
        return []
    shoulder_diff = abs(left[1] - right[1]) / head[1] if head[1] else 1
    if shoulder_diff > PRICE_TOLERANCE * 2:
        return []

    neckline = np.min(high[left[0]:right[0] + 1]) if right[0] > left[0] else (left[1] + right[1]) / 2
    return [_pattern(
        "Omuz-Baş-Omuz", "deneysel", "geometrik",
        "Orta tepe iki yanındaki tepelerden yüksek; klasik dönüş formasyonu adayı.",
        [
            {"kind": "marker", "idx": _abs_idx(0, window_start, left[0]), "price": left[1], "style": "double"},
            {"kind": "marker", "idx": _abs_idx(0, window_start, head[0]), "price": head[1], "style": "double"},
            {"kind": "marker", "idx": _abs_idx(0, window_start, right[0]), "price": right[1], "style": "double"},
            {"kind": "hline", "price": float(neckline), "label": "Boyun çizgisi", "style": "triangle"},
        ],
    )]


def _detect_cup_handle(
    close: np.ndarray,
    troughs: list[tuple[int, float]],
    window_start: int,
) -> list[dict[str, Any]]:
    if len(close) < 60:
        return []

    n = len(close)
    mid = n // 2
    left_rim = float(np.max(close[:mid]))
    cup_slice = close[max(0, mid - 15): min(n, mid + 15)]
    cup_bottom = float(np.min(cup_slice)) if len(cup_slice) else float(np.min(close))
    right_rim = float(np.mean(close[-10:])) if n >= 10 else float(close[-1])
    handle_low = float(np.min(close[-10:]))
    handle_high = float(np.max(close[-10:]))

    cup_depth = (left_rim - cup_bottom) / left_rim if left_rim else 0
    recovery = (right_rim - cup_bottom) / cup_bottom if cup_bottom else 0
    handle_pullback = (handle_high - close[-1]) / handle_high if handle_high else 0

    if not (0.08 < cup_depth < 0.40 and recovery > 0.05 and 0.01 < handle_pullback < 0.15):
        return []

    cup_start = _abs_idx(0, window_start, 0)
    cup_end = _abs_idx(0, window_start, n - 11)
    handle_start = _abs_idx(0, window_start, n - 10)
    handle_end = _abs_idx(0, window_start, n - 1)

    return [_pattern(
        "Çanak-Kulp (Cup & Handle)", "deneysel", "geometrik",
        "U şeklinde çanak ardından kısa geri çekilme (kulp). Deneysel tespit — doğrulama önerilir.",
        [
            {
                "kind": "rect", "style": "cup",
                "start_idx": cup_start, "end_idx": cup_end,
                "price_low": cup_bottom, "price_high": left_rim,
                "label": "Çanak",
            },
            {
                "kind": "rect", "style": "handle",
                "start_idx": handle_start, "end_idx": handle_end,
                "price_low": handle_low, "price_high": handle_high,
                "label": "Kulp",
            },
            {"kind": "hline", "price": left_rim, "label": "Çanak ağzı", "style": "cup"},
        ],
    )]


def _detect_flag(close: np.ndarray, window_start: int) -> list[dict[str, Any]]:
    if len(close) < 30:
        return []

    n = len(close)
    pole_start_local = max(0, n - 25)
    pole_end_local = max(0, n - 15)
    pole_start = close[pole_start_local]
    pole_end = close[pole_end_local]
    flag_section = close[pole_end_local:]

    pole_move = (pole_end - pole_start) / pole_start if pole_start else 0
    flag_range = (
        (np.max(flag_section) - np.min(flag_section)) / np.mean(flag_section)
        if np.mean(flag_section) else 1
    )

    if not (abs(pole_move) > 0.06 and flag_range < 0.07):
        return []

    direction = "Boğa" if pole_move > 0 else "Ayı"
    pole_lo = float(min(pole_start, pole_end))
    pole_hi = float(max(pole_start, pole_end))
    flag_lo = float(np.min(flag_section))
    flag_hi = float(np.max(flag_section))

    return [_pattern(
        f"Flama (Flag) [{direction}]",
        "deneysel", "geometrik",
        "Güçlü hareket (direk) ardından dar konsolidasyon. Kırılım yönü izlenmeli.",
        [
            {
                "kind": "rect", "style": "pole",
                "start_idx": _abs_idx(0, window_start, pole_start_local),
                "end_idx": _abs_idx(0, window_start, pole_end_local),
                "price_low": pole_lo, "price_high": pole_hi,
                "label": "Direk",
            },
            {
                "kind": "rect", "style": "flag",
                "start_idx": _abs_idx(0, window_start, pole_end_local),
                "end_idx": _abs_idx(0, window_start, n - 1),
                "price_low": flag_lo, "price_high": flag_hi,
                "label": "Flama",
            },
        ],
    )]


def _pattern(
    name: str,
    confidence: str,
    category: str,
    description: str,
    regions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "name": name,
        "confidence": confidence,
        "category": category,
        "description": description,
        "regions": regions,
    }


def _slope(x: list[int], y: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])
