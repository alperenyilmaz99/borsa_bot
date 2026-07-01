# BIST100 Teknik Analiz Danışmanı

BIST100 hisselerini teknik göstergelere göre tarayan, her hisseye **üç ayrı vade skoru** (0–100) veren ve isteğe bağlı olarak Claude ile kısa Türkçe gerekçe üreten bir Streamlit uygulaması.

**Temel prensip:** Skorlama %100 Python/mat matematik. LLM yalnızca hazır göstergeleri yorumlar; fiyat tahmini veya al/sat kararı üretmez.

---

## Hızlı Başlangıç

```bash
cd bist-advisor
pip install -r requirements.txt
```

Üst klasörde veya `bist-advisor/` içinde `.env` dosyası:

```env
ANTHROPIC_API_KEY=sk-ant-...
# veya
CLAUDE_API_KEY=sk-ant-...
```

API anahtarı olmadan uygulama çalışır; sadece LLM yorumları devre dışı kalır.

```bash
python -m streamlit run app.py
```

Tarayıcı: `http://localhost:8501`

> Windows'ta `streamlit` komutu PATH'te olmayabilir. Her zaman `python -m streamlit` kullanın.

---

## Mimari

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  app.py     │────▶│ data/fetcher │────▶│  indicators   │────▶│   scorer    │
│ (Streamlit) │     │  (yfinance)  │     │  (ta lib)     │     │ (matematik) │
└──────┬──────┘     └──────────────┘     └───────────────┘     └──────┬──────┘
       │                                                                │
       │         ┌──────────────┐                                       │
       └────────▶│ llm/advisor  │◀── yalnızca top 5 × 3 vade ────────┘
                 │ (Claude API) │
                 └──────────────┘
```

Veri tek yönlü akar: **OHLCV → göstergeler → skorlar → (opsiyonel) LLM yorumu**.

---

## Klasör Yapısı

```
bist-advisor/
├── app.py                 # Streamlit UI — orkestrasyon katmanı
├── config.py              # Ticker listesi, API key, veri parametreleri
├── requirements.txt
├── data/
│   └── fetcher.py         # yfinance veri çekme + cache
├── analysis/
│   ├── indicators.py      # Teknik gösterge hesaplama
│   └── scorer.py          # Üç vadeli skorlama
├── llm/
│   └── advisor.py         # Claude ile gösterge yorumu
└── .streamlit/
    └── config.toml        # headless mod, usage stats kapalı
```

---

## Modül Detayları

### `config.py`

| Sabit | Değer | Açıklama |
|-------|-------|----------|
| `BIST100_TICKERS` | 100 hisse | yfinance formatı: `THYAO.IS` |
| `DATA_PERIOD` | `"1y"` | Son 1 yıl veri |
| `DATA_INTERVAL` | `"1d"` | Günlük mum |
| `LLM_MODEL` | `claude-sonnet-4-6` | LangChain Anthropic modeli |
| `ANTHROPIC_API_KEY` | env | `ANTHROPIC_API_KEY` veya `CLAUDE_API_KEY` |

`.env` hem `bist-advisor/.env` hem de üst klasör `../.env` dosyasından okunur.

---

### `data/fetcher.py`

**Girdi:** ticker string (`"GARAN.IS"`)  
**Çıktı:** `pd.DataFrame` — sütunlar: `Open, High, Low, Close, Volume`

| Fonksiyon | Ne yapar |
|-----------|----------|
| `get_stock_data(ticker)` | Tek hisse çeker. `@st.cache_data(ttl=3600)` ile 1 saat cache |
| `get_all_stocks()` | `BIST100_TICKERS` listesini döngüyle çeker |
| `ticker_symbol(ticker)` | `"THYAO.IS"` → `"THYAO"` |

**Hata toleransı:** Tek hisse hata verirse atlanır, program çökmez. Hisse başına 0.3 sn bekleme (rate limit). Minimum 30 bar olmayan veriler elenir.

---

### `analysis/indicators.py`

**Girdi:** OHLCV DataFrame  
**Çıktı:** `calculate_indicators(df)` → `dict` (son bar değerleri)

Hesaplanan göstergeler (`ta` kütüphanesi):

| Gösterge | Detay |
|----------|-------|
| RSI | 14 periyot |
| MACD | MACD, sinyal, histogram |
| SMA | 20, 50, 200 |
| Hacim oranı | Son hacim / 20 günlük ortalama |
| Fiyat değişimi | 1A (~21), 3A (~63), 6A (~126) işlem günü % |
| SMA pozisyonu | `above_sma_20/50/200` (bool) |
| Cross | `golden_death_cross`: `golden_cross`, `death_cross`, `golden`, `death`, `neutral` |

NaN/eksik veri `_safe_float()` ile güvenli şekilde `None`'a dönüşür.

`add_indicators(df)` grafik için tam sütunlu DataFrame döner (detay sekmesindeki SMA çizgisi).

---

### `analysis/scorer.py`

**Girdi:** OHLCV DataFrame (içeride `calculate_indicators` çağrılır)  
**Çıktı:** `StockScore` dataclass

```python
@dataclass
class StockScore:
    ticker: str
    symbol: str
    short_term_score: float   # 0-100
    mid_term_score: float
    long_term_score: float
    indicators: dict
    short_breakdown: dict     # kriter → katkı puanı
    mid_breakdown: dict
    long_breakdown: dict
```

#### Skor formülü

Her vade için: `skor = Σ (faktör × ağırlık)` — faktörler 0.0–1.0 arası, ağırlıklar toplamı 100.

**Kısa vade** (`SHORT_TERM_WEIGHTS`):

| Kriter | Ağırlık | Faktör mantığı |
|--------|---------|----------------|
| RSI < 40 (aşırı satım) | 30 | RSI 30 altı → 1.0, 30–40 arası kademeli |
| MACD pozitif kesişim | 30 | `bullish_cross` → 1.0, `bullish` → 0.7 |
| Fiyat > SMA20 | 25 | bool |
| Hacim > ortalama | 15 | oran ≥ 1.5 → 1.0, ≥ 1.0 → 0.7 |

**Orta vade** (`MID_TERM_WEIGHTS`):

| Kriter | Ağırlık |
|--------|---------|
| Golden cross aktif | 30 |
| Fiyat > SMA50 | 25 |
| 3A momentum pozitif | 25 |
| Trend gücü | 20 |

**Uzun vade** (`LONG_TERM_WEIGHTS`):

| Kriter | Ağırlık |
|--------|---------|
| Fiyat > SMA200 | 35 |
| 6A getiri pozitif | 35 |
| Uzun trend (SMA50 > SMA200) | 30 |

Ağırlıklar `scorer.py` dosya başında sabit olarak tanımlı — tek satır değiştirerek ayarlanır.

| Fonksiyon | Ne yapar |
|-----------|----------|
| `score_stock(ticker, df)` | Tek hisse, üç skor |
| `score_all_stocks(dict, top_n=5)` | Tüm evreni skorlar; her vade için top N döner |

```python
{
    "short_term": [StockScore, ...],  # top 5
    "mid_term":   [StockScore, ...],
    "long_term":  [StockScore, ...],
    "all":        [StockScore, ...],  # skorlanan tüm hisseler
}
```

---

### `llm/advisor.py`

**LLM burada skor hesaplamaz.** Yalnızca `indicators` dict'ini okuyup 2–3 cümle Türkçe gerekçe yazar.

| Fonksiyon | Ne yapar |
|-----------|----------|
| `generate_commentary(symbol, indicators, score, term)` | Tek hisse + vade için 1 API çağrısı |
| `generate_picks_commentaries(results)` | `score_all_stocks` çıktısındaki 3×top_n hisse için döngü |

**Maliyet sınırı:** Varsayılan top_n=5 → en fazla **15 Claude çağrısı** (5 kısa + 5 orta + 5 uzun). Detay sekmesinde LLM çağrısı yok.

Prompt kısıtları:
- Fiyat tahmini yasak
- "Yükselecek / alın" gibi kesin dil yasak
- Yatırım tavsiyesi değil uyarısı zorunlu

---

### `app.py`

İki sekme:

**Tarama**
1. `get_all_stocks()` → ~100 hisse OHLCV
2. `score_all_stocks(data, top_n)` → üç skor tablosu
3. (Opsiyonel) `generate_picks_commentaries(results)` → hisse başına yorum kartı

**Hisse Detayı**
1. `get_stock_data(selected)` → tek hisse
2. `score_stock(selected, df)` → üç skor + breakdown grafikleri
3. `add_indicators(df)` → SMA fiyat grafiği
4. LLM yok (maliyet kontrolü)

---

## Tipik Çalışma Akışı (kod perspektifi)

```python
# 1. Veri
from data.fetcher import get_all_stocks
data = get_all_stocks()          # dict[str, DataFrame]

# 2. Skor
from analysis.scorer import score_all_stocks
results = score_all_stocks(data, top_n=5)

# 3. En iyi kısa vade hisseleri
for s in results["short_term"]:
    print(s.symbol, s.short_term_score, s.indicators["rsi_14"])

# 4. (Opsiyonel) LLM
from llm.advisor import generate_commentary
text = generate_commentary(
    stock_symbol="THYAO",
    indicators=results["short_term"][0].indicators,
    score=results["short_term"][0].short_term_score,
    term="Kısa Vade",
)
```

---

## Bağımlılıklar

```
streamlit       # UI
yfinance        # BIST verisi (Yahoo Finance)
pandas, numpy   # veri işleme
ta              # teknik analiz göstergeleri
langchain       # LLM orchestration
langchain-anthropic
python-dotenv   # .env yükleme
```

---

## Bilinen Sınırlamalar

- **Veri kaynağı:** Yahoo Finance (`yfinance`). BIST verisi gecikmeli veya eksik olabilir; bazı tickers boş döner ve atlanır.
- **BIST100 listesi:** `config.py` içinde statik. Endeks dönemsel güncellenir; manuel güncelleme gerekir.
- **Skor ≠ tavsiye:** Skorlar kural tabanlı teknik filtre; fundamental analiz, haber, likidite derinliği yok.
- **LLM:** Gösterge yorumu üretir; halüsinasyon riski için prompt kısıtlı tutulmuş, yine de doğrulama kullanıcıya aittir.
- **Cache:** `get_stock_data` Streamlit cache kullanır; aynı oturumda 1 saat içinde yeniden çekilmez.

---

## Geliştirme Notları

| Değişiklik yapmak istiyorsan | Dosya |
|------------------------------|-------|
| Skor ağırlıkları | `analysis/scorer.py` → `*_WEIGHTS` sabitleri |
| Yeni gösterge eklemek | `analysis/indicators.py` → `calculate_indicators` |
| Ticker listesi | `config.py` → `BIST100_TICKERS` |
| LLM prompt / model | `llm/advisor.py`, `config.py` → `LLM_MODEL` |
| Cache süresi | `data/fetcher.py` → `ttl=3600` |
| Rate limit bekleme | `data/fetcher.py` → `FETCH_DELAY_SEC` |

---

## Uyarı

Bu uygulama eğitim ve bilgilendirme amaçlıdır. Üretilen skorlar ve AI yorumları **yatırım tavsiyesi değildir**. Gerçek işlem kararları için profesyonel danışmanlık alın.
