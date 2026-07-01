"""Teknik analiz skorlama — tamamen matematiksel, LLM kullanılmaz."""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.timeframe import DEFAULT_TIMEFRAME
from analysis.indicators import calculate_indicators
from analysis.patterns import detect_patterns
from data.fetcher import ticker_symbol

# --- Kısa vade ağırlıkları (toplam 100) ---
SHORT_TERM_WEIGHTS = {
    "rsi_oversold": 30,        # RSI < 40 → aşırı satım fırsatı
    "macd_bullish_cross": 30,  # MACD pozitif kesişim
    "above_sma_20": 25,        # Fiyat > SMA20
    "volume_above_avg": 15,    # Hacim > 20 günlük ortalama
}

# --- Orta vade ağırlıkları (toplam 100) ---
MID_TERM_WEIGHTS = {
    "golden_cross": 30,        # Golden cross aktif (SMA50 > SMA200)
    "above_sma_50": 25,        # Fiyat > SMA50
    "momentum_3m": 25,         # 3 aylık momentum pozitif
    "trend_strength": 20,      # Trend gücü
}

# --- Uzun vade ağırlıkları (toplam 100) ---
LONG_TERM_WEIGHTS = {
    "above_sma_200": 35,       # Fiyat > SMA200
    "return_6m": 35,          # 6 aylık getiri pozitif
    "long_uptrend": 30,        # Uzun dönem yukarı trend (SMA50 > SMA200)
}

TOP_N_DEFAULT = 5


@dataclass
class StockScore:
    ticker: str
    symbol: str
    short_term_score: float
    mid_term_score: float
    long_term_score: float
    indicators: dict = field(default_factory=dict)
    patterns: list = field(default_factory=list)
    short_breakdown: dict = field(default_factory=dict)
    mid_breakdown: dict = field(default_factory=dict)
    long_breakdown: dict = field(default_factory=dict)

    @property
    def signals(self) -> dict:
        """app.py / advisor.py uyumluluğu."""
        return self.indicators

    @property
    def score(self) -> float:
        """Genel skor: üç vadenin ortalaması."""
        return round(
            (self.short_term_score + self.mid_term_score + self.long_term_score) / 3,
            1,
        )


def score_stock(
    ticker: str,
    df,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> StockScore | None:
    """Tek hisse için üç vadeli skor hesaplar."""
    indicators = calculate_indicators(df, timeframe_key)
    if not indicators:
        return None
    patterns = detect_patterns(df, timeframe_key)
    return score_from_indicators(ticker, indicators, patterns)


def score_from_indicators(
    ticker: str,
    indicators: dict,
    patterns: list | None = None,
) -> StockScore:
    """Hazır gösterge dict'inden üç vadeli skor üretir."""
    short_score, short_bd = _compute_term_score(
        _short_term_factors(indicators), SHORT_TERM_WEIGHTS
    )
    mid_score, mid_bd = _compute_term_score(
        _mid_term_factors(indicators), MID_TERM_WEIGHTS
    )
    long_score, long_bd = _compute_term_score(
        _long_term_factors(indicators), LONG_TERM_WEIGHTS
    )

    return StockScore(
        ticker=ticker,
        symbol=ticker_symbol(ticker),
        short_term_score=short_score,
        mid_term_score=mid_score,
        long_term_score=long_score,
        indicators=indicators,
        patterns=patterns or [],
        short_breakdown=short_bd,
        mid_breakdown=mid_bd,
        long_breakdown=long_bd,
    )


def score_all_stocks(
    stock_data_dict: dict,
    top_n: int = TOP_N_DEFAULT,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> dict[str, list[StockScore]]:
    """Tüm hisseleri skorlar; her vade için en yüksek top_n hisseyi döner."""
    all_scores: list[StockScore] = []

    for ticker, df in stock_data_dict.items():
        result = score_stock(ticker, df, timeframe_key)
        if result:
            all_scores.append(result)

    return {
        "short_term": _top_by(all_scores, "short_term_score", top_n),
        "mid_term": _top_by(all_scores, "mid_term_score", top_n),
        "long_term": _top_by(all_scores, "long_term_score", top_n),
        "all": all_scores,
    }


# --- Kısa vade faktörleri (0.0 – 1.0) ---

def _short_term_factors(ind: dict) -> dict[str, float]:
    return {
        "rsi_oversold": _factor_rsi_oversold(ind.get("rsi_14")),
        "macd_bullish_cross": _factor_macd_bullish(ind.get("macd_signal")),
        "above_sma_20": _factor_bool(ind.get("above_sma_20")),
        "volume_above_avg": _factor_volume(ind.get("volume_ratio")),
    }


def _mid_term_factors(ind: dict) -> dict[str, float]:
    return {
        "golden_cross": _factor_golden_cross(ind.get("golden_death_cross")),
        "above_sma_50": _factor_bool(ind.get("above_sma_50")),
        "momentum_3m": _factor_momentum(ind.get("price_change_3m")),
        "trend_strength": _factor_trend_strength(ind.get("adx_strength")),
    }


def _long_term_factors(ind: dict) -> dict[str, float]:
    return {
        "above_sma_200": _factor_bool(ind.get("above_sma_200")),
        "return_6m": _factor_momentum(ind.get("price_change_6m")),
        "long_uptrend": _factor_long_uptrend(ind),
    }


# --- Faktör hesaplayıcılar ---

def _factor_rsi_oversold(rsi: float | None) -> float:
    if rsi is None:
        return 0.0
    if rsi < 30:
        return 1.0
    if rsi < 40:
        return 0.5 + (40 - rsi) / 10 * 0.5
    return 0.0


def _factor_macd_bullish(macd_label: str | None) -> float:
    if macd_label == "bullish_cross":
        return 1.0
    if macd_label == "bullish":
        return 0.7
    if macd_label == "neutral":
        return 0.3
    return 0.0


def _factor_bool(value: bool | None) -> float:
    if value is True:
        return 1.0
    if value is False:
        return 0.0
    return 0.5


def _factor_volume(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    if ratio >= 1.5:
        return 1.0
    if ratio >= 1.0:
        return 0.7
    if ratio >= 0.8:
        return 0.4
    return 0.0


def _factor_golden_cross(cross: str | None) -> float:
    if cross in ("golden_cross", "golden"):
        return 1.0
    if cross in ("death_cross", "death"):
        return 0.0
    return 0.3


def _factor_momentum(change_pct: float | None) -> float:
    if change_pct is None:
        return 0.0
    if change_pct > 15:
        return 1.0
    if change_pct > 10:
        return 0.85
    if change_pct > 5:
        return 0.7
    if change_pct > 0:
        return 0.55
    if change_pct > -5:
        return 0.25
    return 0.0


def _factor_trend_strength(adx_label: str | None) -> float:
    if adx_label == "strong_trend":
        return 1.0
    if adx_label == "moderate_trend":
        return 0.65
    return 0.3


def _factor_long_uptrend(ind: dict) -> float:
    cross = ind.get("golden_death_cross", "neutral")
    if cross in ("golden_cross", "golden"):
        return 1.0
    if cross in ("death_cross", "death"):
        return 0.0
    sma_50 = ind.get("sma_50")
    sma_200 = ind.get("sma_200")
    if sma_50 is not None and sma_200 is not None and sma_50 > sma_200:
        return 0.75
    return 0.2


def _compute_term_score(
    factors: dict[str, float],
    weights: dict[str, int],
) -> tuple[float, dict[str, float]]:
    breakdown: dict[str, float] = {}
    total = 0.0
    for key, weight in weights.items():
        contribution = factors.get(key, 0.0) * weight
        breakdown[key] = round(contribution, 1)
        total += contribution
    return round(max(0.0, min(100.0, total)), 1), breakdown


def _top_by(scores: list[StockScore], attr: str, n: int) -> list[StockScore]:
    return sorted(scores, key=lambda s: getattr(s, attr), reverse=True)[:n]
