"""Claude ile teknik gösterge yorumu — skorlama burada yapılmaz."""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from analysis.patterns import format_patterns_for_display
from analysis.scorer import StockScore
from config import ANTHROPIC_API_KEY, LLM_MODEL

SYSTEM_PROMPT = """Sen bir borsa teknik analiz asistanısın.

Görevin: Verilen teknik göstergeleri, formasyonları ve skoru yorumlamak.

Kurallar:
- Yalnızca sana verilen göstergeleri ve formasyonları yorumla; yeni veri uydurma
- Fiyat tahmini yapma, hedef fiyat verme
- "Yükselecek", "düşecek", "alın", "satın" gibi kesin ifadeler kullanma
- "deneysel" veya "güvenilmez" işaretli formasyonları kesin gerçek gibi sunma;
  bunlardan bahsederken olasılık dili kullan ("sinyal verebilir", "dikkat çekiyor")
- "kesin" işaretli mum formasyonlarını da yatırım garantisi olarak yorumlama
- 2-3 cümle, sade Türkçe yaz
- Bu bir yatırım tavsiyesi değildir; yalnızca teknik gösterge yorumudur"""

TA_SYSTEM_PROMPT = """Sen disiplinli bir BIST teknik analiz uzmanısın.

Görevin: Python ile hesaplanmış göstergeleri (RSI, MACD, Bollinger, SMA, hacim) okuyup
yapılandırılmış bir teknik analiz raporu yazmak.

Kurallar:
- Yalnızca verilen sayıları ve sinyal etiketlerini kullan; veri uydurma
- Al/Sat/Tut veya kesin yön önerisi verme; "teknik görünüm", "momentum", "trend" dili kullan
- Fiyat hedefi, stop-loss veya getiri tahmini yapma
- RSI: 30 altı aşırı satım, 70 üstü aşırı alım bağlamında yorumla
- MACD: kesişim ve histogram yönünü açıkla
- Bollinger: fiyatın bantlara göre konumunu belirt
- SMA 20/50/200 ve golden/death cross durumunu trend bölümünde değerlendir
- Hacim oranını hareketin güvenilirliği açısından yorumla
- Deneysel formasyonları olasılık diliyle anlat
- Markdown başlıkları kullan (## ile)
- Türkçe, net ve özlü yaz
- Son satırda uyarı: yatırım tavsiyesi değildir"""


_SIGNAL_LABELS = {
    "bullish_cross": "boğa kesişimi",
    "bearish_cross": "ayı kesişimi",
    "bullish": "boğa",
    "bearish": "ayı",
    "neutral": "nötr",
    "oversold": "aşırı satım",
    "overbought": "aşırı alım",
    "high_volume": "yüksek hacim",
    "low_volume": "düşük hacim",
    "normal": "normal",
    "golden_cross": "golden cross (yeni)",
    "death_cross": "death cross (yeni)",
    "golden": "golden cross aktif",
    "death": "death cross aktif",
    "strong_trend": "güçlü trend",
    "moderate_trend": "orta trend",
    "weak_trend": "zayıf trend",
}


def get_advisor(*, max_tokens: int = 300) -> ChatAnthropic | None:
    if not ANTHROPIC_API_KEY:
        return None
    return ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def _fmt_signal(val: str | None) -> str:
    if not val:
        return "veri yok"
    return _SIGNAL_LABELS.get(val, val)


def _format_indicators(indicators: dict) -> str:
    def _fmt(key: str, label: str) -> str:
        val = indicators.get(key)
        if val is None:
            return f"- {label}: veri yok"
        if isinstance(val, float):
            return f"- {label}: {val:.2f}"
        if isinstance(val, bool):
            return f"- {label}: {'evet' if val else 'hayır'}"
        if key.endswith("_signal") or key in ("golden_death_cross", "sma_trend", "adx_strength", "bb_signal"):
            return f"- {label}: {_fmt_signal(str(val))}"
        return f"- {label}: {val}"

    lines = [
        _fmt("close", "Son fiyat (TL)"),
        "",
        "=== Momentum ===",
        _fmt("rsi_14", "RSI (14)"),
        _fmt("rsi_signal", "RSI sinyali"),
        _fmt("macd", "MACD çizgisi"),
        _fmt("macd_signal_line", "MACD sinyal çizgisi"),
        _fmt("macd_hist", "MACD histogram"),
        _fmt("macd_signal", "MACD durumu"),
        "",
        "=== Bollinger Bands (20, 2) ===",
        _fmt("bb_upper", "Üst bant"),
        _fmt("bb_middle", "Orta bant"),
        _fmt("bb_lower", "Alt bant"),
        _fmt("bb_signal", "Bollinger sinyali"),
        "",
        "=== Trend (SMA) ===",
        _fmt("sma_20", "SMA20"),
        _fmt("sma_50", "SMA50"),
        _fmt("sma_200", "SMA200"),
        _fmt("above_sma_20", "Fiyat > SMA20"),
        _fmt("above_sma_50", "Fiyat > SMA50"),
        _fmt("above_sma_200", "Fiyat > SMA200"),
        _fmt("sma_trend", "Kısa-orta trend"),
        _fmt("golden_death_cross", "SMA50/SMA200"),
        _fmt("adx_strength", "Trend gücü (3A momentum)"),
        "",
        "=== Hacim & Getiri ===",
        _fmt("volume_ratio", "Hacim / 20g ortalama"),
        _fmt("volume_signal", "Hacim sinyali"),
        _fmt("price_change_1m", "1 aylık değişim (%)"),
        _fmt("price_change_3m", "3 aylık değişim (%)"),
        _fmt("price_change_6m", "6 aylık değişim (%)"),
    ]
    return "\n".join(lines)


def _format_scores(scores: dict[str, float] | None) -> str:
    if not scores:
        return ""
    return (
        f"- Kısa vade skoru: {scores.get('short', '-')}/100\n"
        f"- Orta vade skoru: {scores.get('mid', '-')}/100\n"
        f"- Uzun vade skoru: {scores.get('long', '-')}/100"
    )


def _format_patterns(patterns: list[dict[str, str]] | None) -> str:
    if not patterns:
        return "Tespit edilen formasyon yok"
    return format_patterns_for_display(patterns)


def _invoke_llm(system: str, user: str, *, max_tokens: int = 300) -> str:
    llm = get_advisor(max_tokens=max_tokens)
    if not llm:
        return "⚠️ ANTHROPIC_API_KEY bulunamadı. `.env` dosyasına `ANTHROPIC_API_KEY=sk-ant-...` ekleyin."
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def generate_technical_analysis(
    stock_symbol: str,
    indicators: dict,
    scores: dict[str, float] | None = None,
    patterns: list[dict[str, str]] | None = None,
) -> str:
    """Hesaplanmış göstergelere dayalı yapılandırılmış teknik analiz raporu."""
    indicator_text = _format_indicators(indicators)
    pattern_text = _format_patterns(patterns)
    score_text = _format_scores(scores)

    user_prompt = f"""Hisse: {stock_symbol} (BIST)
Piyasa: Borsa İstanbul — günlük mum verisi

Hesaplanmış Teknik Göstergeler:
{indicator_text}

Kod Tabanlı Vade Skorları:
{score_text or 'Skor bilgisi yok'}

Tespit Edilen Formasyonlar:
{pattern_text}

Aşağıdaki bölümlerle markdown rapor yaz:

## Genel Teknik Görünüm
(1-2 cümle özet)

## Momentum (RSI & MACD)
(RSI seviyesi ve MACD/histogram yorumu)

## Trend (SMA & Hareketli Ortalamalar)
(SMA20/50/200 konumu, golden/death cross)

## Bollinger & Hacim
(Bant konumu ve hacim oranı)

## Formasyonlar
(Varsa; deneysel olanları temkinli yorumla)

## Vade Perspektifi
(Kısa / orta / uzun vade skorlarına göre genel tablo — al/sat demeden)

Kesin yön önerisi verme. Sadece verilen göstergeleri yorumla."""

    return _invoke_llm(TA_SYSTEM_PROMPT, user_prompt, max_tokens=900)


def generate_commentary(
    stock_symbol: str,
    indicators: dict,
    score: float,
    term: str,
    patterns: list[dict[str, str]] | None = None,
) -> str:
    """Tek hisse + vade için 2-3 cümlelik teknik gerekçe üretir."""
    indicator_text = _format_indicators(indicators)
    pattern_text = _format_patterns(patterns)

    user_prompt = f"""Hisse: {stock_symbol}
Vade: {term}
Teknik Skor: {score}/100

Teknik Göstergeler:
{indicator_text}

Tespit Edilen Formasyonlar:
{pattern_text}

Formasyon notu: [kesin] = TA-Lib mum formasyonu veya SMA kesişimi;
[deneysel] = pivot tabanlı tahmin, güvenilirliği düşük olabilir.

Yukarıdaki göstergelere ve formasyonlara dayanarak {term} perspektifinde
2-3 cümlelik teknik gerekçe yaz. Deneysel formasyonları kesin gerçek gibi sunma.
Sadece verilen verileri yorumla; fiyat tahmini yapma.
Bu bir yatırım tavsiyesi değildir, yalnızca teknik gösterge yorumudur."""

    return _invoke_llm(SYSTEM_PROMPT, user_prompt, max_tokens=300)


def generate_picks_commentaries(scoring_results: dict) -> list[dict]:
    """Skorlamadan seçilen hisseler için yorum üretir (3 vade × top_n)."""
    picks: list[tuple[StockScore, str, str, float]] = [
        (s, s.symbol, "Kısa Vade", s.short_term_score)
        for s in scoring_results.get("short_term", [])
    ]
    picks += [
        (s, s.symbol, "Orta Vade", s.mid_term_score)
        for s in scoring_results.get("mid_term", [])
    ]
    picks += [
        (s, s.symbol, "Uzun Vade", s.long_term_score)
        for s in scoring_results.get("long_term", [])
    ]

    commentaries: list[dict] = []
    for stock, symbol, term, score in picks:
        text = generate_commentary(
            symbol, stock.indicators, score, term, stock.patterns
        )
        commentaries.append({
            "symbol": symbol,
            "term": term,
            "score": score,
            "patterns": stock.patterns,
            "commentary": text,
        })

    return commentaries
