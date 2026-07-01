"""Formasyon işaretli fiyat grafiği — matplotlib."""

from __future__ import annotations

from dataclasses import dataclass, field
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

_MAX_LABEL_LEN = 28


@dataclass
class _LabelSpec:
    anchor_x: float
    anchor_y: float
    text: str
    color: str
    priority: int = 5


@dataclass
class _LabelPlacer:
    """Etiketleri sağ sütunda dikey olarak yerleştirir; çakışmayı önler."""

    ax: Any
    plot_len: int
    specs: list[_LabelSpec] = field(default_factory=list)

    def add(
        self,
        anchor_x: float,
        anchor_y: float,
        text: str,
        color: str,
        *,
        priority: int = 5,
    ) -> None:
        text = _shorten(text)
        if not text:
            return
        if any(s.text == text for s in self.specs):
            return
        self.specs.append(
            _LabelSpec(anchor_x, anchor_y, text, color, priority)
        )

    def render(self) -> None:
        if not self.specs:
            return

        ymin, ymax = self.ax.get_ylim()
        y_range = ymax - ymin
        min_sep = y_range * 0.062
        margin = y_range * 0.02

        specs = sorted(self.specs, key=lambda s: (s.priority, -s.anchor_y))

        placed_y: list[float] = []
        label_x = self.plot_len + max(2, self.plot_len * 0.04)

        for spec in specs:
            target_y = _clamp(spec.anchor_y, ymin + margin, ymax - margin)
            target_y = _resolve_y_collision(target_y, placed_y, min_sep, ymin + margin, ymax - margin)
            placed_y.append(target_y)

            self.ax.annotate(
                spec.text,
                xy=(spec.anchor_x, spec.anchor_y),
                xytext=(label_x, target_y),
                fontsize=6.5,
                color=spec.color,
                fontweight="bold",
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=spec.color,
                    lw=0.7,
                    alpha=0.65,
                    shrinkA=2,
                    shrinkB=2,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor=spec.color,
                    alpha=0.95,
                    linewidth=0.8,
                ),
                clip_on=False,
                zorder=10,
            )

        self.ax.set_xlim(-1, label_x + self.plot_len * 0.22)


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
        2, 1, figsize=(14, 7), sharex=True,
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

    placer = _LabelPlacer(ax_price, plot_len)
    for pat in patterns:
        _draw_pattern_regions(ax_price, pat, x_offset, plot_len, placer)

    ax_price.set_title(
        f"{symbol} — {timeframe_label} · Formasyonlu Grafik" if symbol
        else f"{timeframe_label} · Formasyonlu Grafik",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax_price.set_ylabel("Fiyat (TL)")
    ax_price.grid(True, alpha=0.25)
    ax_price.legend(loc="upper left", fontsize=8, ncol=4)

    placer.render()

    if "Volume" in data.columns:
        colors = [
            "#ef5350" if c < o else "#26a69a"
            for c, o in zip(data["Close"], data["Open"])
        ]
        ax_vol.bar(x, data["Volume"], color=colors, alpha=0.7, width=0.8)
        ax_vol.set_ylabel("Hacim")

    ax_vol.set_xlabel("Zaman (sondan geriye)")
    _set_date_ticks(ax_vol, data.index, plot_len)

    fig.subplots_adjust(hspace=0.08, right=0.98)
    return fig


def _shorten(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_LABEL_LEN:
        return text
    return text[: _MAX_LABEL_LEN - 1] + "…"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _resolve_y_collision(
    y: float,
    placed: list[float],
    min_sep: float,
    ymin: float,
    ymax: float,
) -> float:
    if not placed:
        return y

    for _ in range(40):
        collision = False
        for py in placed:
            if abs(y - py) < min_sep:
                collision = True
                y = py + min_sep
                break
        if not collision:
            break

    if y > ymax:
        y = placed[-1] - min_sep if placed else ymax
        for _ in range(40):
            collision = False
            for py in placed:
                if abs(y - py) < min_sep:
                    collision = True
                    y = py - min_sep
                    break
            if not collision:
                break

    return _clamp(y, ymin, ymax)


def _draw_candlesticks(ax, data: pd.DataFrame) -> None:
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
    placer: _LabelPlacer,
) -> None:
    confidence = pattern.get("confidence", "deneysel")
    edge_color = CONFIDENCE_COLORS.get(confidence, "#ef6c00")
    name = pattern.get("name", "Formasyon")
    priority = 1 if confidence == "kesin" else 3

    anchor_x: float | None = None
    anchor_y: float | None = None
    regions = pattern.get("regions", [])

    for region in regions:
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
            mid_x = (i0 + i1) / 2
            mid_y = (region.get("price_low", 0) + region.get("price_high", 0)) / 2
            anchor_x, anchor_y = mid_x, mid_y

        elif kind == "hline":
            price = region.get("price")
            if price is not None:
                ax.axhline(price, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
                if anchor_x is None:
                    anchor_x = plot_len * 0.85
                    anchor_y = price

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
                if anchor_x is None:
                    anchor_x, anchor_y = plot_pts[-1]

        elif kind == "marker":
            px = _to_plot_idx(region["idx"], x_offset, plot_len)
            if px is not None:
                price = region.get("price", 0)
                ax.scatter([px], [price], color=edge_color, s=60, zorder=5, marker="v", alpha=0.9)
                anchor_x, anchor_y = px, price

    if anchor_x is not None and anchor_y is not None:
        placer.add(anchor_x, anchor_y, name, edge_color, priority=priority)


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
