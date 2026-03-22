"""
Data fetching via yfinance with Streamlit caching (15-min TTL)
"""
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def fetch_quote(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    period: 1wk, 1mo, 3mo, 6mo, 1y, 2y, 5y
    """
    period_map = {
        "1W": "5d", "1M": "1mo", "3M": "3mo",
        "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y"
    }
    yf_period = period_map.get(period, "1y")
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=yf_period, auto_adjust=True)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_financials(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        # Income statement
        income = t.income_stmt  # annual
        cashflow = t.cashflow
        balance = t.balance_sheet

        return {
            "info": info,
            "income": income,
            "cashflow": cashflow,
            "balance": balance,
        }
    except Exception as e:
        return {"error": str(e)}


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
    result["analyst_target"] = info.get("targetMeanPrice")
    result["analyst_low"] = info.get("targetLowPrice")
    result["analyst_high"] = info.get("targetHighPrice")
    result["recommendation"] = info.get("recommendationKey", "").upper()

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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_universe_snapshot(tickers_list: list) -> pd.DataFrame:
    """Fetch basic metrics for a list of (symbol, name) tuples"""
    symbols = [t[0] for t in tickers_list]
    names = {t[0]: t[1] for t in tickers_list}
    rows = []
    for sym in symbols:
        try:
            info = yf.Ticker(sym).info
            price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
            prev = info.get("previousClose") or price
            chg = ((price - prev) / prev * 100) if prev else 0
            rows.append({
                "Ticker": sym,
                "Company": names.get(sym, sym),
                "Price": round(price, 2),
                "Change %": round(chg, 2),
                "Market Cap ($B)": round((info.get("marketCap") or 0) / 1e9, 1),
                "P/E": round(info.get("trailingPE") or 0, 1),
                "52W High": round(info.get("fiftyTwoWeekHigh") or 0, 2),
                "52W Low": round(info.get("fiftyTwoWeekLow") or 0, 2),
                "Beta": round(info.get("beta") or 0, 2),
                "Sector": info.get("sector", "—"),
            })
        except Exception:
            rows.append({
                "Ticker": sym, "Company": names.get(sym, sym),
                "Price": 0, "Change %": 0, "Market Cap ($B)": 0,
                "P/E": 0, "52W High": 0, "52W Low": 0, "Beta": 0, "Sector": "—"
            })
    return pd.DataFrame(rows)
