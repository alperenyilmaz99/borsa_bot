"""BIST100 Teknik Analiz Yatırım Danışmanı — Streamlit arayüzü."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from config import BIST100_TICKERS, ANTHROPIC_API_KEY
import data.fetcher as _fetcher

get_stock_data = getattr(_fetcher, "get_stock_data", None) or _fetcher.fetch_single
get_all_stocks = getattr(_fetcher, "get_all_stocks", None) or _fetcher.fetch_all
ticker_symbol = _fetcher.ticker_symbol
from analysis.patterns import detect_patterns, patterns_summary_table
from analysis.chart_plot import plot_chart_with_patterns
from analysis.scorer import score_all_stocks, score_stock
from analysis.timeframe import DEFAULT_TIMEFRAME, get_timeframe, timeframe_options
from llm.advisor import generate_picks_commentaries, generate_technical_analysis


st.set_page_config(
    page_title="BIST100 Teknik Analiz Danışmanı",
    page_icon="📈",
    layout="wide",
)

st.title("📈 BIST100 Teknik Analiz Danışmanı")
st.caption("Skorlama kod tabanlıdır · LLM yalnızca yorum üretir")

if not ANTHROPIC_API_KEY:
    st.warning("ANTHROPIC_API_KEY tanımlı değil — LLM yorumları devre dışı.")

_tf_options = timeframe_options()
_tf_keys = [k for k, _ in _tf_options]
_tf_labels = dict(_tf_options)

with st.sidebar:
    st.header("⚙️ Ayarlar")
    timeframe_key = st.selectbox(
        "Zaman dilimi (mum periyodu)",
        options=_tf_keys,
        format_func=lambda k: _tf_labels[k],
        index=_tf_keys.index(DEFAULT_TIMEFRAME),
    )
    tf = get_timeframe(timeframe_key)
    st.caption(
        f"Veri: son **{tf.period}** · mum: **{tf.label}** "
        f"(`{tf.interval}`)"
    )
    if tf.note:
        st.info(tf.note)

tab_scan, tab_detail = st.tabs(["🔍 Tarama", "📊 Hisse Detayı"])


def _score_table(scores: list, score_attr: str, change_labels: tuple[str, str, str]) -> pd.DataFrame:
    rows = []
    for s in scores:
        ind = s.indicators
        rows.append({
            "Hisse": s.symbol,
            "Skor": getattr(s, score_attr),
            "Fiyat (TL)": f"{ind.get('close', 0):.2f}",
            "RSI": f"{ind.get('rsi_14', 0):.1f}" if ind.get("rsi_14") else "-",
            change_labels[0]: ind.get("price_change_1m", "-"),
            change_labels[1]: ind.get("price_change_3m", "-"),
            change_labels[2]: ind.get("price_change_6m", "-"),
        })
    return pd.DataFrame(rows)


def _indicator_metrics(ind: dict, change_labels: tuple[str, str, str]) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("RSI (14)", f"{ind['rsi_14']:.1f}" if ind.get("rsi_14") else "-")
    c2.metric("MACD Hist", f"{ind['macd_hist']:.3f}" if ind.get("macd_hist") else "-")
    c3.metric("Hacim Oranı", f"{ind['volume_ratio']:.2f}x" if ind.get("volume_ratio") else "-")
    c4.metric(change_labels[0], ind.get("price_change_1m", "-"))
    c5.metric(change_labels[1], ind.get("price_change_3m", "-"))
    c6.metric(change_labels[2], ind.get("price_change_6m", "-"))


def _show_patterns(patterns: list) -> None:
    st.subheader("📐 Tespit Edilen Formasyonlar")
    if not patterns:
        st.info(
            "Bu periyotta belirgin formasyon tespit edilmedi. "
            "Deneysel formasyonlar (flama, çanak-kulp vb.) yalnızca belirli koşullarda görünür."
        )
        return

    summary = patterns_summary_table(patterns)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    kesin = sum(1 for p in patterns if p.get("confidence") == "kesin")
    deneysel = len(patterns) - kesin
    c1, c2 = st.columns(2)
    c1.metric("Kesin (TA-Lib / matematik)", kesin)
    c2.metric("Deneysel (heuristik)", deneysel)
    st.caption(
        "Yeşil = kesin tespit · Turuncu = deneysel — yanlış pozitif olabilir."
    )


with tab_scan:
    col1, col2 = st.columns(2)
    with col1:
        top_n = st.slider("Vade başına gösterilecek hisse", 3, 15, 5)
    with col2:
        use_llm = st.checkbox("LLM yorumu üret", value=bool(ANTHROPIC_API_KEY))

    if st.button("🚀 Analizi Başlat", type="primary", use_container_width=True):
        with st.spinner(
            f"BIST100 verileri çekiliyor ({tf.label}) ve analiz ediliyor..."
        ):
            data = get_all_stocks(timeframe_key=timeframe_key)
            results = score_all_stocks(data, top_n=top_n, timeframe_key=timeframe_key)

        st.success(
            f"{len(data)} hisse · {len(results['all'])} skorlandı · "
            f"Zaman dilimi: **{tf.label}**"
        )

        if results["all"]:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("⚡ Kısa Vade")
                st.dataframe(
                    _score_table(results["short_term"], "short_term_score", tf.change_labels),
                    use_container_width=True,
                    hide_index=True,
                )
            with c2:
                st.subheader("📅 Orta Vade")
                st.dataframe(
                    _score_table(results["mid_term"], "mid_term_score", tf.change_labels),
                    use_container_width=True,
                    hide_index=True,
                )
            with c3:
                st.subheader("🏔️ Uzun Vade")
                st.dataframe(
                    _score_table(results["long_term"], "long_term_score", tf.change_labels),
                    use_container_width=True,
                    hide_index=True,
                )

            if use_llm:
                total_calls = top_n * 3
                with st.spinner(f"Claude {total_calls} hisse için yorum üretiyor..."):
                    commentaries = generate_picks_commentaries(results)
                st.subheader("🤖 AI Teknik Gerekçeler")
                for item in commentaries:
                    st.markdown(
                        f"**{item['symbol']}** · {item['term']} · Skor: {item['score']}/100\n\n"
                        f"{item['commentary']}"
                    )
                    st.divider()
                st.caption(
                    "⚠️ Bu yorumlar yatırım tavsiyesi değildir; yalnızca teknik gösterge yorumudur."
                )
        else:
            st.info("Skorlanabilir hisse bulunamadı. Farklı zaman dilimi deneyin.")

with tab_detail:
    selected = st.selectbox(
        "Hisse seçin",
        options=sorted(BIST100_TICKERS),
        format_func=ticker_symbol,
    )
    use_llm_detail = st.checkbox(
        "Claude ile teknik analiz raporu",
        value=bool(ANTHROPIC_API_KEY),
        disabled=not ANTHROPIC_API_KEY,
        help="RSI, MACD, Bollinger, SMA ve formasyonları Claude yorumlar.",
    )
    chart_lookback = st.slider(
        f"Grafikte gösterilecek mum sayısı ({tf.lookback_unit})",
        min_value=tf.lookback_min,
        max_value=tf.lookback_max,
        value=tf.lookback_default,
    )

    if st.button("📊 Detaylı Analiz", use_container_width=True):
        with st.spinner(f"{ticker_symbol(selected)} · {tf.label} analiz ediliyor..."):
            df = get_stock_data(selected, timeframe_key=timeframe_key)
            if df.empty:
                st.error(
                    f"Veri çekilemedi. {tf.label} için Yahoo verisi sınırlı olabilir — "
                    "günlük veya haftalık deneyin."
                )
            else:
                result = score_stock(selected, df, timeframe_key)

                if result:
                    st.caption(f"Toplam {len(df)} mum · Gösterilen: son {chart_lookback}")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Kısa Vade", f"{result.short_term_score}/100")
                    m2.metric("Orta Vade", f"{result.mid_term_score}/100")
                    m3.metric("Uzun Vade", f"{result.long_term_score}/100")
                    m4.metric("Fiyat", f"{result.indicators['close']:.2f} TL")

                    st.subheader("Teknik Göstergeler")
                    _indicator_metrics(result.indicators, tf.change_labels)

                    patterns = result.patterns or detect_patterns(df, timeframe_key)
                    _show_patterns(patterns)

                    st.subheader("📈 Formasyonlu Fiyat Grafiği")
                    fig = plot_chart_with_patterns(
                        df,
                        patterns,
                        lookback=chart_lookback,
                        symbol=result.symbol,
                        timeframe_label=tf.label,
                    )
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)

                    st.subheader("Skor Dağılımı")
                    bd_cols = st.columns(3)
                    for col, (title, bd) in zip(
                        bd_cols,
                        [
                            ("Kısa Vade", result.short_breakdown),
                            ("Orta Vade", result.mid_breakdown),
                            ("Uzun Vade", result.long_breakdown),
                        ],
                    ):
                        with col:
                            st.write(f"**{title}**")
                            st.bar_chart(
                                pd.DataFrame(
                                    [{"Gösterge": k, "Katkı": v} for k, v in bd.items()]
                                ).set_index("Gösterge")
                            )

                    if use_llm_detail:
                        with st.spinner("Claude teknik analiz raporu hazırlıyor..."):
                            report = generate_technical_analysis(
                                stock_symbol=result.symbol,
                                indicators=result.indicators,
                                scores={
                                    "short": result.short_term_score,
                                    "mid": result.mid_term_score,
                                    "long": result.long_term_score,
                                },
                                patterns=result.patterns,
                            )
                        st.subheader("🤖 Claude Teknik Analiz Raporu")
                        st.markdown(report)
                        st.caption(
                            "⚠️ Göstergeler Python ile hesaplanır; Claude yalnızca yorumlar. "
                            "Yatırım tavsiyesi değildir."
                        )
                    elif not ANTHROPIC_API_KEY:
                        st.info(
                            "Claude teknik analizi için `.env` dosyasına "
                            "`ANTHROPIC_API_KEY` ekleyin."
                        )
