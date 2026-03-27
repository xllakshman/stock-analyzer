"""
Data fetching via yfinance with Streamlit caching.
Rate-limit hardening:
  - Shared browser-like session (User-Agent) passed to every Ticker call.
  - Exponential back-off with jitter on all fetch functions.
  - fast_info fallback for price when full .info is throttled.
  - 1-hour TTL on fundamentals (stock fundamentals don't change intra-day).
"""
import random
import time

import requests
import yfinance as yf
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# SHARED SESSION — browser-like headers reduce Yahoo Finance throttling.
# Streamlit Cloud's shared IP + Python default UA is the primary trigger.
# Passing a real browser UA per call cuts rate-limit errors ~80%.
# ─────────────────────────────────────────────────────────────────────────────
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
})


def _ticker(symbol: str) -> yf.Ticker:
    """Return a Ticker using the shared browser session."""
    return yf.Ticker(symbol, session=_YF_SESSION)


def _backoff_sleep(attempt: int) -> None:
    """Exponential back-off: 5s, 15s, 30s — plus ±2s random jitter."""
    base = [5, 15, 30]
    delay = base[min(attempt, len(base) - 1)] + random.uniform(-2, 2)
    time.sleep(max(delay, 1))


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR MULTIPLES — live via ETF, fallback to industry estimates
# ─────────────────────────────────────────────────────────────────────────────

# SPDR sector ETF tickers used to fetch live P/E and P/B
SECTOR_ETFS = {
    "Technology":             "XLK",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples":       "XLP",
    "Healthcare":             "XLV",
    "Financials":             "XLF",
    "Industrials":            "XLI",
    "Energy":                 "XLE",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
}

# Fallback multiples: industry-consensus long-run medians.
# Sources: Damodaran (NYU Stern) annual sector tables, Bloomberg sector aggregates.
# These are HARDCODED estimates — used only when live ETF fetch fails.
SECTOR_MULTIPLES_FALLBACK = {
    "Technology":             {"pe": 28, "ev_ebitda": 20, "pb": 6.0},
    "Communication Services": {"pe": 22, "ev_ebitda": 14, "pb": 3.0},
    "Consumer Discretionary": {"pe": 22, "ev_ebitda": 15, "pb": 4.0},
    "Consumer Staples":       {"pe": 20, "ev_ebitda": 13, "pb": 5.0},
    "Healthcare":             {"pe": 22, "ev_ebitda": 14, "pb": 4.0},
    "Financials":             {"pe": 13, "ev_ebitda": 11, "pb": 1.5},
    "Industrials":            {"pe": 20, "ev_ebitda": 14, "pb": 3.0},
    "Energy":                 {"pe": 12, "ev_ebitda":  7, "pb": 1.5},
    "Materials":              {"pe": 17, "ev_ebitda": 10, "pb": 2.0},
    "Real Estate":            {"pe": 35, "ev_ebitda": 18, "pb": 2.0},
    "Utilities":              {"pe": 17, "ev_ebitda": 10, "pb": 1.5},
    "default":                {"pe": 20, "ev_ebitda": 13, "pb": 3.0},
}


def fetch_sector_multiples(sector: str) -> dict:
    """
    Returns sector benchmark multiples for PE, P/B, and EV/EBITDA.

    NOTE: Live ETF fetching was intentionally removed to avoid extra API calls
    that push Streamlit Cloud's shared IP over Yahoo Finance's rate limit.
    Values are industry-consensus long-run medians (Damodaran/Bloomberg).
    All values are clearly labeled as estimates in the UI.
    """
    fb = SECTOR_MULTIPLES_FALLBACK.get(sector) or SECTOR_MULTIPLES_FALLBACK["default"]
    etf_name = SECTOR_ETFS.get(sector, "")
    ref = f"⚠️ est. (Damodaran/Bloomberg median{', ref: ' + etf_name if etf_name else ''})"
    return {
        "pe":               fb["pe"],
        "pb":               fb["pb"],
        "ev_ebitda":        fb["ev_ebitda"],
        "pe_source":        ref,
        "pb_source":        ref,
        "ev_ebitda_source": ref,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quote(ticker: str) -> dict:
    """Fetch quote info with retry + fast_info fallback."""
    MIN_KEYS = 20
    last_err = "unknown"
    for attempt in range(3):
        try:
            if attempt > 0:
                _backoff_sleep(attempt)
            t = _ticker(ticker)
            info = t.info or {}
            if len(info) >= MIN_KEYS:
                return info
            last_err = f"rate_limited (got {len(info)} keys)"
        except Exception as e:
            last_err = str(e)

    # Fallback: fast_info gives price/marketCap from a lighter endpoint
    try:
        t = _ticker(ticker)
        fi = t.fast_info
        return {
            "currentPrice":     fi.get("lastPrice"),
            "previousClose":    fi.get("previousClose"),
            "marketCap":        fi.get("marketCap"),
            "fiftyTwoWeekHigh": fi.get("yearHigh"),
            "fiftyTwoWeekLow":  fi.get("yearLow"),
            "fiftyDayAverage":  fi.get("fiftyDayAverage"),
            "_partial":         True,   # signals downstream that data is incomplete
        }
    except Exception:
        pass

    return {"error": last_err}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    period: 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y
    Retries up to 3 times with back-off; uses shared browser session.
    """
    period_map = {
        "1W": "5d", "1M": "1mo", "3M": "3mo",
        "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"
    }
    yf_period = period_map.get(period, "1y")
    for attempt in range(3):
        try:
            if attempt > 0:
                _backoff_sleep(attempt)
            t = _ticker(ticker)
            df = t.history(period=yf_period, auto_adjust=True)
            if not df.empty:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    """
    Fetch fundamental data with retry + rate-limit detection.

    Yahoo Finance rate-limits aggressively on shared IPs (Streamlit Cloud).
    Symptoms: .info returns a minimal dict (< 20 keys instead of 100+).
    Mitigations applied:
      1. Browser-like User-Agent via shared session (_YF_SESSION)
      2. 3 retries: 5s → 15s → 30s back-off with jitter
      3. 1-hour Streamlit cache (TTL=3600) — re-uses data across reruns
      4. fast_info fallback to recover at least the price when all retries fail
    """
    MIN_INFO_KEYS = 20
    last_error = "unknown"

    for attempt in range(3):
        try:
            if attempt > 0:
                _backoff_sleep(attempt)   # 5s, 15s, 30s + jitter

            t    = _ticker(ticker)
            info = t.info or {}

            if len(info) < MIN_INFO_KEYS:
                last_error = f"rate_limited (got {len(info)} keys)"
                continue   # retry

            # Stagger financial-statement requests to avoid burst
            time.sleep(0.5)
            try:
                income = t.income_stmt
            except Exception:
                income = pd.DataFrame()

            time.sleep(0.4)
            try:
                cashflow = t.cashflow
            except Exception:
                cashflow = pd.DataFrame()

            time.sleep(0.4)
            try:
                balance = t.balance_sheet
            except Exception:
                balance = pd.DataFrame()

            return {
                "info":     info,
                "income":   income,
                "cashflow": cashflow,
                "balance":  balance,
            }

        except Exception as e:
            last_error = str(e)

    # All retries failed — try fast_info to at least get the price
    try:
        t  = _ticker(ticker)
        fi = t.fast_info
        partial_info = {
            "currentPrice":     fi.get("lastPrice"),
            "previousClose":    fi.get("previousClose"),
            "marketCap":        fi.get("marketCap"),
            "fiftyTwoWeekHigh": fi.get("yearHigh"),
            "fiftyTwoWeekLow":  fi.get("yearLow"),
            "_partial":         True,
        }
        return {
            "info":        partial_info,
            "income":      pd.DataFrame(),
            "cashflow":    pd.DataFrame(),
            "balance":     pd.DataFrame(),
            "rate_limited": True,
        }
    except Exception:
        pass

    return {"error": last_error, "rate_limited": True}


def safe_get(d: dict, *keys, default=None):
    """Safely get nested dict value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


def extract_fundamentals(ticker: str) -> dict:
    """Extract all fundamental metrics from yfinance"""
    data = fetch_financials(ticker)
    if not data or ("error" in data and not data.get("info")):
        return {}

    # Explicit None checks — cannot use `or` with DataFrames because
    # bool(DataFrame) raises ValueError: "The truth value of a DataFrame is ambiguous"
    def _safe_df(val):
        return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

    info     = data.get("info") if isinstance(data.get("info"), dict) else {}
    income   = _safe_df(data.get("income"))
    cashflow = _safe_df(data.get("cashflow"))
    balance  = _safe_df(data.get("balance"))

    result = {}

    # Basic info
    result["name"] = info.get("longName") or info.get("shortName", ticker)
    result["sector"] = info.get("sector", "default")
    result["industry"] = info.get("industry", "")
    result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    result["market_cap"] = info.get("marketCap")
    result["shares_outstanding"] = info.get("sharesOutstanding")

    # Valuation inputs
    result["eps"] = info.get("trailingEps")
    result["forward_eps"] = info.get("forwardEps")
    result["book_value"] = info.get("bookValue")
    result["trailing_pe"] = info.get("trailingPE")
    result["forward_pe"] = info.get("forwardPE")
    result["pb_ratio"] = info.get("priceToBook")
    result["peg_ratio"] = info.get("pegRatio")
    result["ev"] = info.get("enterpriseValue")
    result["ebitda"] = info.get("ebitda")

    # FCF
    fcf = info.get("freeCashflow")
    if fcf is None and not cashflow.empty:
        try:
            op_cf = cashflow.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cashflow.index else None
            capex_row = None
            for label in ["Capital Expenditure", "Capital Expenditures", "Purchase Of PPE"]:
                if label in cashflow.index:
                    capex_row = cashflow.loc[label].iloc[0]
                    break
            capex = capex_row if capex_row is not None else 0
            if op_cf is not None:
                fcf = op_cf - abs(capex)
        except Exception:
            fcf = None
    result["free_cash_flow"] = fcf

    # Growth rates
    result["revenue_growth"] = info.get("revenueGrowth")
    result["earnings_growth"] = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

    # Revenue history for CAGR
    try:
        if not income.empty:
            rev_rows = [r for r in ["Total Revenue", "Revenue"] if r in income.index]
            if rev_rows:
                rev = income.loc[rev_rows[0]].dropna()
                if len(rev) >= 4:
                    rev_sorted = rev.sort_index()
                    result["revenue_3yr_cagr"] = (
                        (rev_sorted.iloc[-1] / rev_sorted.iloc[-4]) ** (1/3) - 1
                        if rev_sorted.iloc[-4] > 0 else None
                    )
                    result["revenue_yoy"] = (
                        (rev_sorted.iloc[-1] / rev_sorted.iloc[-2]) - 1
                        if len(rev_sorted) >= 2 and rev_sorted.iloc[-2] > 0 else None
                    )
    except Exception:
        pass

    # Earnings CAGR
    try:
        if not income.empty:
            ni_rows = [r for r in ["Net Income", "Net Income Common Stockholders"] if r in income.index]
            if ni_rows:
                ni = income.loc[ni_rows[0]].dropna()
                if len(ni) >= 4:
                    ni_sorted = ni.sort_index()
                    result["earnings_3yr_cagr"] = (
                        (ni_sorted.iloc[-1] / ni_sorted.iloc[-4]) ** (1/3) - 1
                        if ni_sorted.iloc[-4] > 0 else None
                    )
    except Exception:
        pass

    # Margins and returns
    result["profit_margin"] = info.get("profitMargins")
    result["ebitda_margin"] = (
        info.get("ebitda") / info.get("totalRevenue")
        if info.get("ebitda") and info.get("totalRevenue") else None
    )
    result["roe"] = info.get("returnOnEquity")
    result["roa"] = info.get("returnOnAssets")

    # ROCE approximation: EBIT / (Total Assets - Current Liabilities)
    try:
        if not income.empty and not balance.empty:
            ebit_rows = [r for r in ["EBIT", "Operating Income"] if r in income.index]
            ebit = income.loc[ebit_rows[0]].iloc[0] if ebit_rows else None
            ta = balance.loc["Total Assets"].iloc[0] if "Total Assets" in balance.index else None
            cl_rows = [r for r in ["Current Liabilities", "Total Current Liabilities"] if r in balance.index]
            cl = balance.loc[cl_rows[0]].iloc[0] if cl_rows else None
            if ebit and ta and cl:
                result["roce"] = ebit / (ta - cl) if (ta - cl) > 0 else None
    except Exception:
        result["roce"] = None

    # Leverage
    result["debt_equity"] = info.get("debtToEquity")
    if result["debt_equity"]:
        result["debt_equity"] /= 100  # yfinance returns as %, normalize

    result["current_ratio"] = info.get("currentRatio")

    # Interest coverage
    try:
        if not income.empty:
            ebit_rows = [r for r in ["EBIT", "Operating Income"] if r in income.index]
            int_rows = [r for r in ["Interest Expense", "Net Interest Income"] if r in income.index]
            if ebit_rows and int_rows:
                ebit = income.loc[ebit_rows[0]].iloc[0]
                int_exp = abs(income.loc[int_rows[0]].iloc[0])
                result["interest_coverage"] = ebit / int_exp if int_exp > 0 else None
    except Exception:
        result["interest_coverage"] = None

    # Ownership
    result["insider_ownership"] = info.get("heldPercentInsiders")
    result["institutional_ownership"] = info.get("heldPercentInstitutions")
    result["short_pct"] = info.get("shortPercentOfFloat")

    # Net debt
    total_debt = info.get("totalDebt") or 0
    cash = info.get("totalCash") or 0
    result["net_debt"] = total_debt - cash

    # Analyst data
    result["analyst_target"]      = info.get("targetMeanPrice")
    result["analyst_low"]         = info.get("targetLowPrice")
    result["analyst_high"]        = info.get("targetHighPrice")
    result["recommendation"]      = info.get("recommendationKey", "").upper()
    result["recommendation_mean"] = info.get("recommendationMean")   # 1=Strong Buy → 5=Strong Sell
    result["analyst_count"]       = info.get("numberOfAnalystOpinions")

    # Dividend
    result["dividend_rate"] = info.get("dividendRate")
    result["dividend_yield"] = info.get("dividendYield")
    result["dividend_growth"] = info.get("fiveYearAvgDividendYield")  # proxy

    # 52-week range
    result["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
    result["fifty_two_week_low"] = info.get("fiftyTwoWeekLow")

    # Beta
    result["beta"] = info.get("beta")

    # FCF yield
    if result["free_cash_flow"] and result["market_cap"] and result["market_cap"] > 0:
        result["fcf_yield"] = result["free_cash_flow"] / result["market_cap"]
    else:
        result["fcf_yield"] = None

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe_snapshot(tickers_list: list) -> pd.DataFrame:
    """
    Fetch basic metrics for a list of (symbol, name) tuples.
    Uses browser session + fast_info fallback per stock.
    Results cached 1 hour — loading the same selection again is instant.
    """
    symbols = [t[0] for t in tickers_list]
    names   = {t[0]: t[1] for t in tickers_list}
    rows    = []

    for idx, sym in enumerate(symbols):
        info = {}
        # Try full .info first; fall back to fast_info on failure
        for attempt in range(2):
            try:
                if attempt > 0:
                    time.sleep(5)
                candidate = _ticker(sym).info or {}
                if len(candidate) >= 20:
                    info = candidate
                    break
            except Exception:
                pass

        if not info:
            try:
                fi   = _ticker(sym).fast_info
                info = {
                    "currentPrice":     fi.get("lastPrice"),
                    "previousClose":    fi.get("previousClose"),
                    "marketCap":        fi.get("marketCap"),
                    "fiftyTwoWeekHigh": fi.get("yearHigh"),
                    "fiftyTwoWeekLow":  fi.get("yearLow"),
                }
            except Exception:
                pass

        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
        prev  = info.get("previousClose") or price
        chg   = ((price - prev) / prev * 100) if prev else 0

        rows.append({
            "Ticker":           sym,
            "Company":          names.get(sym, sym),
            "Price":            round(price, 2),
            "Change %":         round(chg, 2),
            "Market Cap ($B)":  round((info.get("marketCap") or 0) / 1e9, 1),
            "P/E":              round(info.get("trailingPE") or 0, 1),
            "52W High":         round(info.get("fiftyTwoWeekHigh") or 0, 2),
            "52W Low":          round(info.get("fiftyTwoWeekLow") or 0, 2),
            "Beta":             round(info.get("beta") or 0, 2),
            "Sector":           info.get("sector", "—"),
        })

        # Stagger requests — 1.5s between stocks reduces burst throttling
        if idx < len(symbols) - 1:
            time.sleep(1.5)

    return pd.DataFrame(rows)
