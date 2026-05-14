"""
financial_tools.py
==================
LangChain tools for financial asset analysis.

Available tools:
    - search_ticker          : Search tickers by name
    - get_ticker_values      : Return historical closing price and volume
    - calculate_indicators   : Compute RSI, EMA, and annualized volatility
    - get_ticker_fundamentals: Return fundamentals (P/E, P/B, DY, Debt, ROE)
    - web_search             : Web search via DuckDuckGo
    - url_to_markdown        : Convert a web page or PDF to plain text
    - predict_price_movement : Train XGBoost and predict next-day price direction
"""

from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from ddgs import DDGS
from langchain.tools import tool
from markitdown import MarkItDown
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_ticker_period(raw: str, default_period: str = "1y") -> tuple[str, str]:
    """
    Parse an input string in the format ``'TICKER PERIOD'``.

    Parameters
    ----------
    raw:
        Raw input string, e.g. ``'BTC-USD 5d'`` or ``'ITUB4.SA 1mo'``.
    default_period:
        Period to use when none is provided.

    Returns
    -------
    tuple[str, str]
        ``(ticker, period)`` with the period normalized to lowercase.
    """
    parts = raw.strip().strip("'\"").split()
    ticker = parts[0]
    period = parts[1].lower() if len(parts) > 1 else default_period
    return ticker, period


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    Parameters
    ----------
    close:
        Series of closing prices.
    window:
        Smoothing window size (default: 14).

    Returns
    -------
    pd.Series
        RSI values in the range 0-100.
    """
    diff = np.diff(close.values)
    gains  = np.insert(np.where(diff > 0,  diff,        0.0), 0, 0.0)
    losses = np.insert(np.where(diff < 0, np.abs(diff), 0.0), 0, 0.0)

    avg_gain = pd.Series(gains).rolling(window=window).mean()
    avg_loss = pd.Series(losses).rolling(window=window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_ema(close: pd.Series, span: int = 80) -> pd.Series:
    """
    Compute the Exponential Moving Average (EMA).

    Parameters
    ----------
    close:
        Series of closing prices.
    span:
        EMA period (default: 80).

    Returns
    -------
    pd.Series
        EMA values.
    """
    return close.ewm(span=span, min_periods=span, adjust=False).mean()


def _compute_volatility(close: pd.Series, window: int = 80) -> pd.Series:
    """
    Compute annualized volatility based on log-returns.

    Parameters
    ----------
    close:
        Series of closing prices.
    window:
        Rolling window size (default: 80).

    Returns
    -------
    pd.Series
        Annualized volatility values.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=window).std() * (365 ** 0.5)


def _download_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance and flatten the multi-level column index.

    Parameters
    ----------
    ticker:
        Asset symbol, e.g. ``'ITUB4.SA'``, ``'BTC-USD'``.
    period:
        Data period, e.g. ``'1y'``, ``'6mo'``, ``'10y'``.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.

    Raises
    ------
    ValueError
        If the download returns an empty DataFrame.
    """
    df = yf.download(ticker, period=period, threads=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")
    df.columns = df.columns.droplevel(1)
    return df.ffill().dropna()


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

_HTML_TAGS_TO_STRIP = ["header", "footer", "nav", "aside", "script", "style", "iframe"]


@tool
def search_ticker(name: str) -> str:
    """
    Search for tickers by asset name and return the 3 most relevant matches.

    Parameters
    ----------
    name:
        Asset or company name to search for, e.g. ``'Apple'``, ``'Bitcoin'``.

    Returns
    -------
    str
        Formatted list of up to 3 matching tickers.
    """
    results = yf.Search(name).quotes
    return str(results[:3])


@tool
def get_ticker_values(ticker_and_period: str) -> str:
    """
    Return the closing price and volume of an asset over a given period.

    Parameters
    ----------
    ticker_and_period:
        String in the format ``'TICKER PERIOD'``.
        Examples: ``'BTC-USD 5d'``, ``'ITUB4.SA 1mo'``, ``'XP 3d'``.
        Valid periods: number + ``d`` / ``mo`` / ``y`` (e.g. ``5d``, ``3mo``, ``1y``).

    Returns
    -------
    str
        Table with ``Close`` and ``Volume`` columns for the requested period.
    """
    ticker, period = _parse_ticker_period(ticker_and_period, default_period="5d")
    df = _download_ohlcv(ticker, period)
    return df[["Close", "Volume"]].to_string()


@tool
def calculate_indicators(ticker_and_period: str) -> str:
    """
    Compute RSI (14), EMA-80, and annualized volatility (80) for an asset.

    Returns the last 5 rows of the computed indicators.

    Parameters
    ----------
    ticker_and_period:
        String in the format ``'TICKER PERIOD'``.
        Examples: ``'BTC-USD 1y'``, ``'ITUB4.SA 2y'``, ``'XP 6mo'``.
        Minimum recommended period: 1 year (``1y``).

    Returns
    -------
    str
        Table with columns ``RSI``, ``EMA80``, and ``Volatility``
        for the last 5 available trading days.
    """
    ticker, period = _parse_ticker_period(ticker_and_period, default_period="1y")
    df = _download_ohlcv(ticker, period)

    result = pd.DataFrame({
        "RSI":        _compute_rsi(df["Close"]).values,
        "EMA80":      _compute_ema(df["Close"], span=80).values,
        "Volatility": _compute_volatility(df["Close"], window=80).values,
    }).dropna().tail(5)

    return result.to_string()


@tool
def get_ticker_fundamentals(ticker: str) -> str:
    """
    Return the key fundamental indicators of a listed asset.

    Collected metrics (when available):
        - **P/E**            : Price-to-Earnings (trailing)
        - **P/B**            : Price-to-Book Value
        - **Dividend Yield** : Annual dividend yield
        - **Debt/Equity**    : Total debt relative to equity
        - **ROE**            : Return on Equity

    Parameters
    ----------
    ticker:
        Asset symbol on Yahoo Finance, e.g. ``'ITUB4.SA'``, ``'AAPL'``.

    Returns
    -------
    str
        Serialized dictionary with the collected fundamentals.
        ``None`` values indicate data not available for the asset.
    """
    ticker = ticker.strip().strip("'\"") 
    info = yf.Ticker(ticker).info

    fundamentals: dict[str, float | None] = {
        "P/E":            info.get("trailingPE"),
        "P/B":            info.get("priceToBook"),
        "Dividend Yield": info.get("dividendYield"),
        "Debt/Equity":    info.get("debtToEquity"),
        "ROE":            info.get("returnOnEquity"),
    }

    for key, value in fundamentals.items():
        print(f"{key}: {value}")

    return str(fundamentals)


@tool
def web_search(query: str) -> str:
    """
    Perform a web search via DuckDuckGo and return the top 5 results.

    Parameters
    ----------
    query:
        Search query text, e.g. ``'Apple Q4 2024 earnings'``.

    Returns
    -------
    str
        List of results, each containing a title, URL, and snippet.
    """
    results = DDGS().text(query, max_results=5)
    return str(results)


@tool
def url_to_markdown(url: str) -> str:
    """
    Convert the content of a URL (HTML page or PDF) into clean plain text.

    For HTML pages, non-essential elements (header, footer, nav, scripts, etc.)
    are removed before conversion. For PDFs, the byte stream is converted directly.

    Parameters
    ----------
    url:
        Address of the web page or PDF file to process.

    Returns
    -------
    str
        Extracted text in Markdown format, or an error message if the
        request or conversion fails.
    """
    md = MarkItDown()
    url = url.strip().strip("'\"") 
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in _HTML_TAGS_TO_STRIP:
                for element in soup.find_all(tag):
                    element.decompose()

            main_content = soup.find("main") or soup.find("article") or soup.body
            html_clean = str(main_content) if main_content else str(soup)
            return md.convert_string(html_clean, extension=".html").text_content

        else:
            ext = ".pdf" if url.lower().endswith(".pdf") else ""
            return md.convert_stream(BytesIO(response.content), extension=ext).text_content

    except Exception as exc:
        return f"Error processing {url}: {exc}"


@tool
def predict_price_movement(ticker_and_period: str) -> str:
    """
    Train an XGBoost classifier on historical data and predict whether the
    asset price will rise or fall on the next trading day.

    Features used:
        - 14-period RSI
        - 80-period EMA
        - 80-period annualized volatility

    Target:
        ``1`` if the next-day return is positive, ``0`` otherwise.

    Parameters
    ----------
    ticker_and_period:
        String in the format ``'TICKER PERIOD'``.
        Examples: ``'BTC-USD 10y'``, ``'ITUB4.SA 5y'``, ``'XP 10y'``.
        Minimum recommended period: 10 years (``10y``) for reliable results.

    Returns
    -------
    str
        Sentence with the predicted direction (``UP`` or ``DOWN``) and
        the model's confidence for that prediction.

    Notes
    -----
    The model is trained and evaluated on the full in-sample dataset for
    internal reference only. The final prediction is made on the last
    available day in the data.
    """
    ticker, period = _parse_ticker_period(ticker_and_period, default_period="10y")
    df = _download_ohlcv(ticker, period)

    forward_return = (df["Close"].shift(-1) - df["Close"]) / df["Close"] * 100
    target = np.where(forward_return > 0, 1, 0)

    features = pd.DataFrame({
        "RSI":        _compute_rsi(df["Close"]).values,
        "EMA80":      _compute_ema(df["Close"], span=80).values,
        "Volatility": _compute_volatility(df["Close"], window=80).values,
        "Target":     target,
    }).dropna()

    X = features.drop("Target", axis=1).values
    y = features["Target"].values

    # Train on all rows except the last one (which is the prediction target)
    model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X[:-1], y[:-1])

    # In-sample accuracy (informational only)
    in_sample_accuracy = accuracy_score(y[:-1], model.predict(X[:-1]))
    print(f"In-sample accuracy: {in_sample_accuracy:.2%}")

    # Predict next trading day direction
    prediction: int = model.predict(X[-1:, :])[0]
    confidence: float = model.predict_proba(X[-1:, :])[0][1]
    direction = "UP" if prediction == 1 else "DOWN"

    return f"Predicted direction: {direction} with {confidence:.1%} confidence."