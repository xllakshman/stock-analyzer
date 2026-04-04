"""
Data fetching via yfinance with Streamlit caching.

yfinance 1.x uses curl_cffi with Chrome impersonation internally.
Do NOT pass a custom requests.Session — yfinance will reject it.
Let yfinance manage its own session for best Yahoo Finance compatibility.

Rate-limit hardening:
  - yfinance 1.x handles cookie/crumb auth and Chrome impersonation natively.
  - fast_info fallback for price when full .info is throttled.
  - 1-hour TTL on fundamentals (stock fundamentals don't change intra-day).
"""
import random
import time

import yfinance as yf
from yfinance.exceptions import YFRateLimitError, YFDataException
import pandas as pd
import streamlit as st


def _ticker(symbol: str) -> yf.Ticker:
    """Return a Ticker. No custom session — yfinance 1.x manages curl_cffi internally."""
    return yf.Ticker(symbol)


def _backoff_sleep(attempt: int) -> None:
    """Back-off: 3s, 8s, 15s — plus ±1s random jitter. Shorter than before for UX."""
    base = [3, 8, 15]
    delay = base[min(attempt, len(base) - 1)] + random.uniform(-1, 1)
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
    """Fetch quote info with retry + fast_info merge + fallback."""
    MIN_KEYS = 10
    last_err = "unknown"
    for attempt in range(3):
        try:
            if attempt > 0:
                _backoff_sleep(attempt)
            t = _ticker(ticker)
            info = t.info or {}
            if len(info) >= MIN_KEYS:
                return _merge_fast_info(info, t)
            last_err = f"sparse_response (got {len(info)} keys)"
        except (YFRateLimitError, YFDataException) as e:
            last_err = f"yf_error: {type(e).__name__}"
        except Exception as e:
            last_err = str(e)[:80]

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
            "_partial":         True,
        }
    except Exception:
        pass

    return {"error": last_err}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    period: 1W, 1M, 3M, 6M, 1Y, 2Y, 5Y
    Retries up to 3 times with back-off.
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
        except (YFRateLimitError, YFDataException):
            pass
        except Exception:
            pass
    return pd.DataFrame()


def _merge_fast_info(info: dict, t: yf.Ticker) -> dict:
    """
    yfinance 1.x retired many keys from .info (currentPrice, marketCap,
    fiftyTwoWeekHigh/Low, previousClose, sharesOutstanding, etc.) and moved
    them to fast_info. This function injects them back so downstream code
    that uses the old key names keeps working.
    """
    try:
        fi = t.fast_info
        retired_map = {
            "currentPrice":          fi.get("lastPrice"),
            "regularMarketPrice":    fi.get("lastPrice"),
            "previousClose":         fi.get("previousClose"),
            "marketCap":             fi.get("marketCap"),
            "fiftyTwoWeekHigh":      fi.get("yearHigh"),
            "fiftyTwoWeekLow":       fi.get("yearLow"),
            "fiftyDayAverage":       fi.get("fiftyDayAverage"),
            "twoHundredDayAverage":  fi.get("twoHundredDayAverage"),
            "sharesOutstanding":     fi.get("shares"),
            "averageVolume":         fi.get("threeMonthAverageVolume"),
        }
        for k, v in retired_map.items():
            if v is not None:
                info.setdefault(k, v)   # don't overwrite if .info already has it
    except Exception:
        pass
    return info


def _fetch_statements(t: yf.Ticker) -> tuple:
    """
    Fetch income statement, cash flow, and balance sheet independently.
    Uses query2.finance.yahoo.com/ws/fundamentals-timeseries — a different
    endpoint from quoteSummary, so it can succeed even when .info is rate-limited.
    Returns (income_df, cashflow_df, balance_df) — any may be empty on failure.
    """
    income = cashflow = balance = pd.DataFrame()
    try:
        income = t.income_stmt
    except Exception:
        pass
    try:
        cashflow = t.cashflow
    except Exception:
        pass
    try:
        balance = t.balance_sheet
    except Exception:
        pass
    return income, cashflow, balance


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    """
    Fetch fundamental data with two independent paths:

    PATH A — .info (quoteSummary via query1): company metadata, P/E, EPS,
      margins, analyst targets, etc. Prone to rate-limiting on shared IPs.

    PATH B — financial statements (fundamentals-timeseries via query2):
      income stmt, cash flow, balance sheet. Different endpoint — more
      rate-limit resistant. Fetched independently so it succeeds even when
      PATH A fails.

    Both paths are attempted; extract_fundamentals() derives missing fields
    from statements when .info fields are None.
    """
    MIN_INFO_KEYS = 10   # rate-limited response has 0-5 keys; real one has 10+

    t = _ticker(ticker)

    # ── PATH A: .info (retry up to 3 times) ─────────────────────
    info = {}
    for attempt in range(3):
        try:
            if attempt > 0:
                _backoff_sleep(attempt)
            raw = t.info or {}
            if len(raw) >= MIN_INFO_KEYS:
                info = raw
                break
        except (YFRateLimitError, YFDataException):
            pass
        except Exception:
            pass

    # Always inject retired price/market fields from fast_info
    info = _merge_fast_info(info, t)

    # ── PATH B: financial statements (independent endpoint) ──────
    income, cashflow, balance = _fetch_statements(t)

    # If BOTH paths returned nothing useful, signal that to the caller
    has_info  = len(info) >= MIN_INFO_KEYS
    has_stmts = not income.empty or not cashflow.empty or not balance.empty
    has_price = bool(info.get("currentPrice") or info.get("regularMarketPrice"))

    if not has_price:
        # Complete failure — nothing usable
        return {"error": "no_data", "rate_limited": True}

    return {
        "info":         info,
        "income":       income,
        "cashflow":     cashflow,
        "balance":      balance,
        "rate_limited": not has_info,   # True when only partial .info was available
    }


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
    # currentPrice / marketCap / sharesOutstanding were retired from .info in yfinance 1.x
    # _merge_fast_info() in fetch_financials injects them back, so they should be present.
    result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
    result["market_cap"] = info.get("marketCap")
    result["shares_outstanding"] = info.get("sharesOutstanding")

    # 52-week range and moving averages — also retired from .info in yfinance 1.x, injected back
    result["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh")
    result["fifty_two_week_low"]  = info.get("fiftyTwoWeekLow")

    # ── Helper: safe row lookup from a statement DataFrame ───────
    def _stmt_val(df: pd.DataFrame, *row_names) -> float | None:
        """Return most-recent value of the first matching row in a statement df."""
        if df is None or df.empty:
            return None
        for name in row_names:
            if name in df.index:
                try:
                    col = df.loc[name].dropna()
                    return float(col.iloc[0]) if len(col) > 0 else None
                except Exception:
                    pass
        return None

    # ── Valuation inputs — prefer .info, fall back to statements ─
    # EPS: yfinance 1.x uses camelCase keys in income_stmt
    _eps_info = info.get("trailingEps")
    _eps_stmt = _stmt_val(income, "DilutedEPS", "BasicEPS",
                          "NormalizedDilutedEPS", "NormalizedBasicEPS")
    result["eps"] = _eps_info if _eps_info is not None else _eps_stmt

    result["forward_eps"] = info.get("forwardEps")

    # Shares — needed to compute book value per share
    _shares = (result["shares_outstanding"] or
               _stmt_val(income, "DilutedAverageShares", "BasicAverageShares") or
               _stmt_val(balance, "OrdinarySharesNumber", "ShareIssued"))
    result["shares_outstanding"] = _shares

    # Book Value Per Share: prefer .info; else CommonStockEquity / shares
    _bv_info = info.get("bookValue")
    if _bv_info is None and _shares and _shares > 0:
        _equity = _stmt_val(balance, "CommonStockEquity", "StockholdersEquity",
                            "TotalEquityGrossMinorityInterest")
        _bv_info = (_equity / _shares) if _equity else None
    result["book_value"] = _bv_info

    result["trailing_pe"] = info.get("trailingPE")
    result["forward_pe"]  = info.get("forwardPE")
    result["pb_ratio"]    = info.get("priceToBook")
    result["peg_ratio"]   = info.get("pegRatio")

    # EV — hard to compute without .info; use what we have
    result["ev"] = info.get("enterpriseValue")
    # If .info has no EV but we have debt/cash from balance sheet, estimate it
    if result["ev"] is None and result["market_cap"]:
        _total_debt  = _stmt_val(balance, "TotalDebt", "LongTermDebt") or 0
        _cash        = _stmt_val(balance, "CashCashEquivalentsAndShortTermInvestments",
                                 "CashAndCashEquivalents") or 0
        result["ev"] = result["market_cap"] + _total_debt - _cash

    # EBITDA — yfinance income_stmt has "EBITDA" row directly
    _ebitda_info = info.get("ebitda")
    _ebitda_stmt = _stmt_val(income, "EBITDA", "NormalizedEBITDA")
    result["ebitda"] = _ebitda_info if _ebitda_info is not None else _ebitda_stmt

    # FCF — cashflow stmt has "FreeCashFlow" row directly
    _fcf_info = info.get("freeCashflow")
    _fcf_stmt = _stmt_val(cashflow, "FreeCashFlow")
    if _fcf_stmt is None:
        # Compute: Operating CF - CapEx
        _ocf   = _stmt_val(cashflow, "OperatingCashFlow", "CashFlowFromContinuingOperatingActivities")
        _capex = _stmt_val(cashflow, "CapitalExpenditure", "PurchaseOfPPE", "CapitalExpenditureReported")
        _fcf_stmt = (_ocf - abs(_capex)) if (_ocf is not None and _capex is not None) else _ocf
    result["free_cash_flow"] = _fcf_info if _fcf_info is not None else _fcf_stmt

    # Total Revenue — for margin computation fallback
    _total_revenue = (info.get("totalRevenue") or
                      _stmt_val(income, "TotalRevenue", "OperatingRevenue"))

    # Growth rates
    result["revenue_growth"]  = info.get("revenueGrowth")
    result["earnings_growth"] = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")

    # Revenue history for CAGR
    try:
        if not income.empty:
            rev_rows = [r for r in ["TotalRevenue", "Total Revenue", "OperatingRevenue"] if r in income.index]
            if rev_rows:
                rev = income.loc[rev_rows[0]].dropna()
                if len(rev) >= 2:
                    rev_sorted = rev.sort_index()
                    if len(rev_sorted) >= 4 and rev_sorted.iloc[-4] > 0:
                        result["revenue_3yr_cagr"] = (
                            (rev_sorted.iloc[-1] / rev_sorted.iloc[-4]) ** (1/3) - 1
                        )
                    if rev_sorted.iloc[-2] > 0:
                        result["revenue_yoy"] = (rev_sorted.iloc[-1] / rev_sorted.iloc[-2]) - 1
                    # Also fill revenue_growth if .info didn't provide it
                    if result["revenue_growth"] is None and "revenue_yoy" in result:
                        result["revenue_growth"] = result.get("revenue_yoy")
    except Exception:
        pass

    # Earnings CAGR
    try:
        if not income.empty:
            ni_rows = [r for r in ["NetIncome", "Net Income", "NetIncomeCommonStockholders",
                                   "NetIncomeFromContinuingOperationNetMinorityInterest"] if r in income.index]
            if ni_rows:
                ni = income.loc[ni_rows[0]].dropna()
                if len(ni) >= 4:
                    ni_sorted = ni.sort_index()
                    if ni_sorted.iloc[-4] > 0:
                        result["earnings_3yr_cagr"] = (
                            (ni_sorted.iloc[-1] / ni_sorted.iloc[-4]) ** (1/3) - 1
                        )
    except Exception:
        pass

    # Margins and returns
    _profit_margin_info = info.get("profitMargins")
    if _profit_margin_info is None and _total_revenue and _total_revenue > 0:
        _net_income = _stmt_val(income, "NetIncome", "NetIncomeCommonStockholders",
                                "NetIncomeFromContinuingOperationNetMinorityInterest")
        _profit_margin_info = (_net_income / _total_revenue) if _net_income is not None else None
    result["profit_margin"] = _profit_margin_info
    # EBITDA margin — use statement EBITDA + revenue as fallback
    _ebitda_for_margin = result["ebitda"]
    _rev_for_margin    = _total_revenue
    result["ebitda_margin"] = (
        _ebitda_for_margin / _rev_for_margin
        if _ebitda_for_margin and _rev_for_margin and _rev_for_margin > 0 else None
    )

    # ROE / ROA — prefer .info; else compute from statements
    _roe_info = info.get("returnOnEquity")
    if _roe_info is None:
        _ni     = _stmt_val(income, "NetIncome", "NetIncomeCommonStockholders",
                            "NetIncomeFromContinuingOperationNetMinorityInterest")
        _equity = _stmt_val(balance, "CommonStockEquity", "StockholdersEquity")
        _roe_info = (_ni / _equity) if (_ni is not None and _equity and _equity > 0) else None
    result["roe"] = _roe_info

    _roa_info = info.get("returnOnAssets")
    if _roa_info is None:
        _ni = _stmt_val(income, "NetIncome", "NetIncomeCommonStockholders",
                        "NetIncomeFromContinuingOperationNetMinorityInterest")
        _ta = _stmt_val(balance, "TotalAssets")
        _roa_info = (_ni / _ta) if (_ni is not None and _ta and _ta > 0) else None
    result["roa"] = _roa_info

    # ROCE: EBIT / (Total Assets - Current Liabilities)
    try:
        _ebit = _stmt_val(income, "EBIT", "OperatingIncome")
        _ta   = _stmt_val(balance, "TotalAssets")
        _cl   = _stmt_val(balance, "CurrentLiabilities")
        if _ebit and _ta and _cl and (_ta - _cl) > 0:
            result["roce"] = _ebit / (_ta - _cl)
        else:
            result["roce"] = None
    except Exception:
        result["roce"] = None

    # Leverage
    _de = info.get("debtToEquity")
    if _de is not None:
        # yfinance returns D/E as % (e.g. 150 = 1.5x); guard against pre-normalized decimals
        result["debt_equity"] = _de / 100 if _de > 20 else _de
    else:
        # Derive from balance sheet: TotalDebt / CommonStockEquity
        _td     = _stmt_val(balance, "TotalDebt", "LongTermDebt")
        _equity = _stmt_val(balance, "CommonStockEquity", "StockholdersEquity")
        result["debt_equity"] = (_td / _equity) if (_td is not None and _equity and _equity > 0) else None

    result["current_ratio"] = info.get("currentRatio")

    # Interest coverage
    try:
        _ebit    = _stmt_val(income, "EBIT", "OperatingIncome")
        _int_exp = _stmt_val(income, "InterestExpense", "InterestExpenseNonOperating",
                             "NetInterestIncome")
        if _ebit and _int_exp and abs(_int_exp) > 0:
            result["interest_coverage"] = _ebit / abs(_int_exp)
        else:
            result["interest_coverage"] = None
    except Exception:
        result["interest_coverage"] = None

    # Ownership
    result["insider_ownership"] = info.get("heldPercentInsiders")
    result["institutional_ownership"] = info.get("heldPercentInstitutions")
    result["short_pct"] = info.get("shortPercentOfFloat")

    # Net debt — prefer .info; else compute from balance sheet
    _total_debt_info = info.get("totalDebt")
    _cash_info       = info.get("totalCash")
    if _total_debt_info is not None:
        result["net_debt"] = (_total_debt_info or 0) - (_cash_info or 0)
    else:
        _td   = _stmt_val(balance, "TotalDebt", "LongTermDebt") or 0
        _cash = _stmt_val(balance, "CashCashEquivalentsAndShortTermInvestments",
                          "CashAndCashEquivalents") or 0
        result["net_debt"] = _td - _cash

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

        # Small stagger between stocks to reduce burst throttling
        if idx < len(symbols) - 1:
            time.sleep(0.5)

    return pd.DataFrame(rows)
