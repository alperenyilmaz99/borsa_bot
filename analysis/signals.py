"""Al / Sat / Tut sinyali — çoklu gösterge oylaması (kod tabanlı)."""

from __future__ import annotations

from dataclasses import dataclass

SIGNAL_STRONG_BUY = "GÜÇLÜ AL"
SIGNAL_BUY = "AL"
SIGNAL_HOLD = "TUT"
SIGNAL_SELL = "SAT"
SIGNAL_STRONG_SELL = "GÜÇLÜ SAT"

SIGNAL_ORDER = [
    SIGNAL_STRONG_SELL,
    SIGNAL_SELL,
    SIGNAL_HOLD,
    SIGNAL_BUY,
    SIGNAL_STRONG_BUY,
]


@dataclass
class SignalResult:
    signal: str
    score: float
    bullish: int
    bearish: int
    neutral: int
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "score": self.score,
            "bullish": self.bullish,
            "bearish": self.bearish,
            "neutral": self.neutral,
            "reasons": self.reasons,
        }


def build_trade_signals(
    indicators: dict,
    short_score: float,
    mid_score: float,
    long_score: float,
) -> dict[str, dict]:
    """Genel + üç vade için sinyal üretir."""
    overall = _indicator_consensus(indicators)
    term_avg = (short_score + mid_score + long_score) / 3
    blended = round(overall.score * 0.55 + (term_avg - 50) * 0.9, 1)
    overall_signal = _numeric_to_signal(blended)
    overall.reasons = overall.reasons[:4]
    overall_dict = overall.to_dict()
    overall_dict["signal"] = overall_signal
    overall_dict["score"] = blended

    return {
        "overall": overall_dict,
        "short_term": _term_signal(short_score, "Kısa vade").to_dict(),
        "mid_term": _term_signal(mid_score, "Orta vade").to_dict(),
        "long_term": _term_signal(long_score, "Uzun vade").to_dict(),
    }


def signal_for_term(trade_signals: dict, term_key: str) -> str:
    return trade_signals.get(term_key, {}).get("signal", SIGNAL_HOLD)


def _term_signal(score: float, label: str) -> SignalResult:
    signal = _score_to_signal(score)
    reasons = [f"{label} teknik skoru: {score}/100"]
    if score >= 58:
        reasons.append("Gösterge seti bu vadede olumlu ağırlıkta")
    elif score <= 42:
        reasons.append("Gösterge seti bu vadede olumsuz ağırlıkta")
    else:
        reasons.append("Göstergeler karışık — net yön zayıf")
    return SignalResult(
        signal=signal,
        score=score,
        bullish=0,
        bearish=0,
        neutral=0,
        reasons=reasons,
    )


def _indicator_consensus(indicators: dict) -> SignalResult:
    votes: list[tuple[float, str | None]] = []

    votes.append(_vote_rsi(indicators))
    votes.append(_vote_macd(indicators))
    votes.append(_vote_sma_trend(indicators))
    votes.append(_vote_bollinger(indicators))
    votes.append(_vote_golden_cross(indicators))
    votes.append(_vote_sma_stack(indicators))
    votes.append(_vote_momentum(indicators))
    votes.append(_vote_volume(indicators))

    weights = [1.2, 1.3, 1.0, 0.9, 1.1, 1.0, 0.8, 0.5]
    total_w = 0.0
    weighted = 0.0
    bullish = bearish = neutral = 0
    reasons: list[str] = []

    for (vote, reason), w in zip(votes, weights):
        if vote > 0.15:
            bullish += 1
        elif vote < -0.15:
            bearish += 1
        else:
            neutral += 1
        weighted += vote * w
        total_w += w
        if reason and abs(vote) >= 0.4:
            reasons.append(reason)

    score = round((weighted / total_w) * 50, 1) if total_w else 0.0
    score = max(-50.0, min(50.0, score))

    return SignalResult(
        signal=_numeric_to_signal(score),
        score=score,
        bullish=bullish,
        bearish=bearish,
        neutral=neutral,
        reasons=reasons[:6],
    )


def _vote_rsi(ind: dict) -> tuple[float, str | None]:
    label = ind.get("rsi_signal", "neutral")
    rsi = ind.get("rsi_14")
    if label == "oversold":
        return 1.0, f"RSI aşırı satım ({rsi:.0f})" if rsi else "RSI aşırı satım"
    if label == "overbought":
        return -1.0, f"RSI aşırı alım ({rsi:.0f})" if rsi else "RSI aşırı alım"
    if label == "bullish":
        return 0.35, None
    if label == "bearish":
        return -0.35, None
    return 0.0, None


def _vote_macd(ind: dict) -> tuple[float, str | None]:
    label = ind.get("macd_signal", "neutral")
    if label == "bullish_cross":
        return 1.0, "MACD pozitif kesişim"
    if label == "bearish_cross":
        return -1.0, "MACD negatif kesişim"
    if label == "bullish":
        return 0.6, "MACD boğa bölgesinde"
    if label == "bearish":
        return -0.6, "MACD ayı bölgesinde"
    return 0.0, None


def _vote_sma_trend(ind: dict) -> tuple[float, str | None]:
    label = ind.get("sma_trend", "neutral")
    if label == "bullish":
        return 0.8, "Fiyat > SMA20 > SMA50 (yükseliş dizilimi)"
    if label == "bearish":
        return -0.8, "Fiyat < SMA20 < SMA50 (düşüş dizilimi)"
    return 0.0, None


def _vote_bollinger(ind: dict) -> tuple[float, str | None]:
    label = ind.get("bb_signal", "neutral")
    if label == "oversold":
        return 0.7, "Fiyat Bollinger alt bandına yakın"
    if label == "overbought":
        return -0.7, "Fiyat Bollinger üst bandına yakın"
    if label == "bullish":
        return 0.3, None
    if label == "bearish":
        return -0.3, None
    return 0.0, None


def _vote_golden_cross(ind: dict) -> tuple[float, str | None]:
    cross = ind.get("golden_death_cross", "neutral")
    if cross in ("golden_cross", "golden"):
        return 0.9, "Golden cross / SMA50 > SMA200"
    if cross in ("death_cross", "death"):
        return -0.9, "Death cross / SMA50 < SMA200"
    return 0.0, None


def _vote_sma_stack(ind: dict) -> tuple[float, str | None]:
    score = 0.0
    parts: list[str] = []
    if ind.get("above_sma_20") is True:
        score += 0.35
        parts.append("SMA20 üstü")
    elif ind.get("above_sma_20") is False:
        score -= 0.35
    if ind.get("above_sma_200") is True:
        score += 0.35
        parts.append("SMA200 üstü")
    elif ind.get("above_sma_200") is False:
        score -= 0.35
    reason = ", ".join(parts) if parts else None
    return max(-1.0, min(1.0, score)), reason


def _vote_momentum(ind: dict) -> tuple[float, str | None]:
    ch3 = ind.get("price_change_3m")
    if ch3 is None:
        return 0.0, None
    if ch3 > 10:
        return 0.7, f"3A momentum güçlü (+{ch3}%)"
    if ch3 > 3:
        return 0.4, None
    if ch3 < -10:
        return -0.7, f"3A momentum zayıf ({ch3}%)"
    if ch3 < -3:
        return -0.4, None
    return 0.0, None


def _vote_volume(ind: dict) -> tuple[float, str | None]:
    label = ind.get("volume_signal", "normal")
    ratio = ind.get("volume_ratio")
    if label == "high_volume" and ind.get("price_change_1m", 0) not in (None, 0):
        ch = ind.get("price_change_1m", 0) or 0
        if ch > 0:
            return 0.5, f"Hacim ortalamanın üstünde ({ratio:.1f}x)" if ratio else "Yüksek hacim"
        if ch < 0:
            return -0.4, "Yüksek hacimle düşüş"
    return 0.0, None


def _score_to_signal(score: float) -> str:
    if score >= 72:
        return SIGNAL_STRONG_BUY
    if score >= 58:
        return SIGNAL_BUY
    if score >= 42:
        return SIGNAL_HOLD
    if score >= 28:
        return SIGNAL_SELL
    return SIGNAL_STRONG_SELL


def _numeric_to_signal(score: float) -> str:
    """-50..+50 konsensüs skorunu sinyale çevirir."""
    if score >= 28:
        return SIGNAL_STRONG_BUY
    if score >= 12:
        return SIGNAL_BUY
    if score > -12:
        return SIGNAL_HOLD
    if score > -28:
        return SIGNAL_SELL
    return SIGNAL_STRONG_SELL
