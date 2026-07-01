"""Formasyon işaretli fiyat grafiği — matplotlib."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.figure import Figure

from analysis.indicators import add_indicators

CONFIDENCE_COLORS = {
    "kesin": "#2e7d32",
    "deneysel": "#ef6c00",
}

REGION_COLORS = {
    "pole": "#42a5f5",
    "flag": "#ffca28",
    "cup": "#ab47bc",
    "handle": "#ce93d8",
    "triangle": "#26a69a",
    "double": "#5c6bc0",
    "marker": "#e53935",
    "trend": "#66bb6a",
}


def plot_chart_with_patterns(
    df: pd.DataFrame,
    patterns: list[dict[str, Any]],
    *,
    lookback: int = 120,
    symbol: str = "",
    timeframe_label: str = "Günlük",
) -> Figure:
    """OHLCV + SMA grafiği üzerine tespit edilen formasyonları çizer."""
    if df is None or df.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.text(0.5, 0.5, "Veri yok", ha="center", va="center")
        return fig

    enriched = add_indicators(df)
    total_len = len(enriched)
    start = max(0, total_len - lookback)
    data = enriched.iloc[start:].copy()
    plot_len = len(data)
    x_offset = start

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    x = range(plot_len)
    ax_price.plot(x, data["Close"], color="#1565c0", linewidth=1.8, label="Kapanış", zorder=3)
    for col, color, lbl in [
        ("SMA_20", "#ff9800", "SMA20"),
        ("SMA_50", "#9c27b0", "SMA50"),
        ("SMA_200", "#795548", "SMA200"),
    ]:
        if col in data.columns:
            ax_price.plot(x, data[col], color=color, linewidth=1.0, alpha=0.85, label=lbl)

    _draw_candlesticks(ax_price, data)

    label_positions: list[tuple[float, float]] = []
    for pat in patterns:
        _draw_pattern_regions(ax_price, pat, x_offset, plot_len, label_positions)

    ax_price.set_title(
        f"{symbol} — {timeframe_label} · Formasyonlu Grafik" if symbol
        else f"{timeframe_label} · Formasyonlu Grafik",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax_price.set_ylabel("Fiyat (TL)")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8, ncol=4)

    if "Volume" in data.columns:
        colors = [
            "#ef5350" if c < o else "#26a69a"
            for c, o in zip(data["Close"], data["Open"])
        ]
        ax_vol.bar(x, data["Volume"], color=colors, alpha=0.7, width=0.8)
        ax_vol.set_ylabel("Hacim")

    ax_vol.set_xlabel("Zaman (sondan geriye)")
    _set_date_ticks(ax_vol, data.index, plot_len)

    fig.subplots_adjust(hspace=0.08)
    return fig


def _draw_candlesticks(ax, data: pd.DataFrame) -> None:
    """Basit mum çubukları (gövde + fitil)."""
    for i, row in enumerate(data.itertuples()):
        o, h, l, c = row.Open, row.High, row.Low, row.Close
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, alpha=0.5, zorder=1)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), (h - l) * 0.02)
        rect = mpatches.Rectangle(
            (i - 0.3, body_bottom), 0.6, body_height,
            facecolor=color, edgecolor=color, alpha=0.35, zorder=2,
        )
        ax.add_patch(rect)


def _to_plot_idx(abs_idx: int, x_offset: int, plot_len: int) -> int | None:
    rel = abs_idx - x_offset
    if 0 <= rel < plot_len:
        return rel
    return None


def _draw_pattern_regions(
    ax,
    pattern: dict[str, Any],
    x_offset: int,
    plot_len: int,
    label_positions: list[tuple[float, float]],
) -> None:
    confidence = pattern.get("confidence", "deneysel")
    edge_color = CONFIDENCE_COLORS.get(confidence, "#ef6c00")
    name = pattern.get("name", "Formasyon")

    for region in pattern.get("regions", []):
        kind = region.get("kind")
        color = REGION_COLORS.get(region.get("style", kind), "#ef6c00")
        alpha = 0.18 if kind in ("rect", "pole", "flag", "cup", "handle") else 0.9

        if kind == "rect":
            i0 = _to_plot_idx(region["start_idx"], x_offset, plot_len)
            i1 = _to_plot_idx(region["end_idx"], x_offset, plot_len)
            if i0 is None and i1 is None:
                continue
            i0 = max(0, i0 if i0 is not None else 0)
            i1 = min(plot_len - 1, i1 if i1 is not None else plot_len - 1)
            ax.axvspan(i0, i1, color=color, alpha=alpha, zorder=0)
            mid_y = (region.get("price_low", 0) + region.get("price_high", 0)) / 2
            _add_label(ax, (i0 + i1) / 2, mid_y, region.get("label", ""), label_positions, edge_color)

        elif kind == "hline":
            price = region.get("price")
            if price is not None:
                ax.axhline(price, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
                _add_label(ax, plot_len * 0.92, price, region.get("label", ""), label_positions, edge_color)

        elif kind == "line":
            pts = region.get("points", [])
            plot_pts = []
            for abs_x, y in pts:
                px = _to_plot_idx(int(abs_x), x_offset, plot_len)
                if px is not None:
                    plot_pts.append((px, y))
            if len(plot_pts) >= 2:
                xs, ys = zip(*plot_pts)
                ax.plot(xs, ys, color=color, linewidth=1.5, linestyle="--", alpha=0.85)
                _add_label(ax, plot_pts[-1][0], plot_pts[-1][1], region.get("label", ""), label_positions, edge_color)

        elif kind == "marker":
            px = _to_plot_idx(region["idx"], x_offset, plot_len)
            if px is not None:
                price = region.get("price", 0)
                ax.scatter([px], [price], color=edge_color, s=80, zorder=5, marker="v")
                _add_label(ax, px, price, name, label_positions, edge_color)


def _add_label(
    ax,
    x: float,
    y: float,
    text: str,
    positions: list[tuple[float, float]],
    color: str,
) -> None:
    if not text:
        return
    for px, py in positions:
        if abs(px - x) < 8 and abs(py - y) < (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03:
            y += (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.annotate(
        text, (x, y), fontsize=7, color=color, fontweight="bold",
        ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.85),
    )
    positions.append((x, y))


def _set_date_ticks(ax, index: pd.Index, plot_len: int) -> None:
    step = max(1, plot_len // 8)
    ticks = list(range(0, plot_len, step))
    if plot_len - 1 not in ticks:
        ticks.append(plot_len - 1)
    labels = []
    for t in ticks:
        try:
            labels.append(index[t].strftime("%d.%m") if hasattr(index[t], "strftime") else str(t))
        except Exception:
            labels.append(str(t))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
