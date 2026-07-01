"""BIST100 hisseleri için yfinance veri çekici."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from analysis.timeframe import DEFAULT_TIMEFRAME, get_timeframe
from config import BIST100_TICKERS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FETCH_DELAY_SEC = 0.3
CACHE_TTL_SEC = 3600

_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _cache_key(ticker: str, interval: str, period: str) -> str:
    return f"{ticker}|{interval}|{period}"


def get_stock_data(
    ticker: str,
    *,
    interval: str | None = None,
    period: str | None = None,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> pd.DataFrame:
    """yfinance ile OHLCV verisi çeker (zaman dilimine göre cache)."""
    tf = get_timeframe(timeframe_key)
    interval = interval or tf.interval
    period = period or tf.period
    min_bars = tf.min_bars

    key = _cache_key(ticker, interval, period)
    now = time.time()
    cached = _cache.get(key)
    if cached and now - cached[0] < CACHE_TTL_SEC:
        return cached[1].copy()

    df = _download_ohlcv(ticker, interval=interval, period=period, min_bars=min_bars)
    _cache[key] = (now, df)
    return df.copy()


def get_all_stocks(
    tickers: list[str] | None = None,
    delay: float = FETCH_DELAY_SEC,
    *,
    timeframe_key: str = DEFAULT_TIMEFRAME,
) -> dict[str, pd.DataFrame]:
    """Config'deki tüm BIST100 hisselerini çeker; hatalı olanları atlar."""
    tf = get_timeframe(timeframe_key)
    tickers = tickers or BIST100_TICKERS
    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for i, ticker in enumerate(tickers):
        if i > 0 and delay > 0:
            time.sleep(delay)

        try:
            df = get_stock_data(ticker, timeframe_key=timeframe_key)
        except Exception as exc:
            logger.warning("%s: beklenmeyen hata — %s", ticker, exc)
            failed.append(ticker)
            continue

        if df.empty:
            logger.warning("%s: veri gelmedi veya yetersiz", ticker)
            failed.append(ticker)
            continue

        if len(df) < tf.min_bars:
            logger.warning(
                "%s: yetersiz bar (%d < %d)", ticker, len(df), tf.min_bars
            )
            failed.append(ticker)
            continue

        result[ticker] = df

    if failed:
        logger.info(
            "%d/%d hisse başarıyla çekildi, atlanan: %s",
            len(result),
            len(tickers),
            ", ".join(failed),
        )
    else:
        logger.info("%d/%d hisse başarıyla çekildi", len(result), len(tickers))

    return result


def clear_cache() -> None:
    _cache.clear()


def _download_ohlcv(
    ticker: str,
    *,
    interval: str,
    period: str,
    min_bars: int,
) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.warning("%s: indirme hatası — %s", ticker, exc)
        return pd.DataFrame()

    return _normalize_ohlcv(df, ticker, min_bars=min_bars)


def _normalize_ohlcv(df: pd.DataFrame, ticker: str, *, min_bars: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.title)
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("%s: eksik sütunlar — %s", ticker, missing)
        return pd.DataFrame()

    cleaned = df[OHLCV_COLUMNS].dropna()
    if len(cleaned) < min_bars:
        return pd.DataFrame()

    return cleaned


def ticker_symbol(ticker: str) -> str:
    """THYAO.IS -> THYAO"""
    return ticker.replace(".IS", "")


fetch_single = get_stock_data
fetch_all = get_all_stocks

__all__ = [
    "get_stock_data",
    "get_all_stocks",
    "ticker_symbol",
    "fetch_single",
    "fetch_all",
    "clear_cache",
]
