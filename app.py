"""
Stock Analyzer — Streamlit App
Search any US or India stock by ticker or company name.
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import time

from utils.tickers import UNIVERSE_MAP
from utils.data import fetch_history, extract_fundamentals, fetch_sector_multiples
from utils.calculations import (
    graham_number, dcf_valuation, pe_based_valuation,
    ev_ebit_valuation, fcf_yield_valuation,
    ev_ebitda_valuation, pb_valuation, peg_signal,
    dividend_discount_model, composite_fair_value,
    fundamental_signal, fundamental_score,
    technical_score, investment_decision, tranche_plan,
    find_support_resistance, StockDecisionEngine,
)
from utils.charts import (
    price_chart, rsi_chart, macd_chart,
    bb_width_chart, atr_chart, gauge_chart,
)

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# RISK APPETITE PRESETS  (institutional-grade weights)
#
# Slot keys:
#   ev_ebit   = EV/EBIT (operating earnings multiple — PE/HF anchor)
#   dcf       = Two-stage 10yr DCF with CAPM WACC
#   pe        = Forward P/E × sector median
#   ev        = EV/EBITDA × sector median
#   fcf_yield = FCF Yield-implied price (required return method)
#   ddm       = Dividend Discount Model (Gordon Growth)
#
# Aggressive  → highest composite (EV/EBITDA + forward PE dominate)
# Moderate    → balanced institutional mix
# Conservative→ lowest composite (EV/EBIT + DCF + FCF Yield anchor)
# ──────────────────────────────────────────────────────────────
RISK_PRESETS = {
    "Aggressive": {
        "desc": (
            "Growth-focused. EV/EBITDA 40% + Forward P/E 35% + Two-Stage DCF 25%. "
            "Excludes operating earnings (EV/EBIT) and FCF Yield — they penalise high-capex "
            "growth companies. 20% DCF growth rate. Suited for tech, cloud, and platform stocks."
        ),
        "weights":    {"ev_ebit": 0,  "dcf": 25, "pe": 35, "ev": 40, "fcf_yield": 0,  "ddm": 0},
        "dcf_growth": 20,
    },
    "Moderate": {
        "desc": (
            "Institutional balanced. EV/EBITDA 35% + Two-Stage DCF 30% + Forward P/E 20% "
            "+ FCF Yield 15%. Reflects a typical sell-side analyst blended approach. 12% DCF growth."
        ),
        "weights":    {"ev_ebit": 0,  "dcf": 30, "pe": 20, "ev": 35, "fcf_yield": 15, "ddm": 0},
        "dcf_growth": 12,
    },
    "Conservative": {
        "desc": (
            "Safety-first / PE-firm style. EV/EBIT 25% + Two-Stage DCF 35% + P/E 20% "
            "+ FCF Yield 15% + DDM 5%. EV/EBIT and FCF Yield are the most conservative "
            "anchors — they strip D&A and demand a return threshold. 8% DCF growth rate."
        ),
        "weights":    {"ev_ebit": 25, "dcf": 35, "pe": 20, "ev": 0,  "fcf_yield": 15, "ddm": 5},
        "dcf_growth": 8,
    },
    "Custom": {
        "desc": "Set your own weights and growth rate with the controls below.",
        "weights":    None,
        "dcf_growth": None,
    },
}

# Human-readable weight key names
WEIGHT_LABELS = {
    "ev_ebit":   "EV / Operating Earnings",
    "dcf":       "DCF (Two-Stage)",
    "pe":        "Forward P/E",
    "ev":        "EV / EBITDA",
    "fcf_yield": "FCF Yield",
    "ddm":       "Dividends (DDM)",
}

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS — dark theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f172a; color: #f1f5f9; }
  [data-testid="stSidebar"] { background-color: #1e293b; }
  [data-testid="stSidebar"] .stMarkdown { color: #f1f5f9; }

  .metric-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
  }
  .metric-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { color: #f1f5f9; font-size: 1.35rem; font-weight: 600; margin-top: 2px; }
  .metric-sub   { color: #64748b; font-size: 0.78rem; margin-top: 2px; }

  .badge-buy    { background:#15803d; color:#dcfce7; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-hold   { background:#854d0e; color:#fef9c3; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-sell   { background:#7f1d1d; color:#fee2e2; padding:3px 10px; border-radius:4px; font-weight:600; }

  .header-bar {
    background: #1e293b; border-radius: 8px; padding: 12px 20px;
    margin-bottom: 16px; border-left: 4px solid #3b82f6;
    display: flex; justify-content: space-between; align-items: center;
  }
  .takeaway-box {
    background: #1e3a5f; border: 1px solid #2563eb; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 14px;
    font-size: 0.88rem; color: #cbd5e1;
  }

  /* Tab navigation — clean underline style */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 2px solid #1e293b;
    border-radius: 0;
    padding: 0;
    gap: 0;
    margin-bottom: 16px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-weight: 500;
    font-size: 0.92rem;
    background: transparent !important;
    border-radius: 0;
    padding: 10px 22px;
    border-bottom: 3px solid transparent;
    margin-bottom: -2px;
    transition: color 0.15s;
    letter-spacing: 0.01em;
  }
  .stTabs [data-baseweb="tab"]:hover { color: #cbd5e1 !important; }
  .stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #f1f5f9 !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #3b82f6 !important;
  }

  .stDataFrame { background: #1e293b !important; }
  .stTextInput input, .stSelectbox select { background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
  .stNumberInput input { background: #1e293b !important; color: #f1f5f9 !important; }
  hr { border-color: #334155; }
  .stProgress > div > div { background-color: #3b82f6 !important; }
  .risk-flag { background:#7f1d1d; color:#fca5a5; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .ok-flag   { background:#14532d; color:#86efac; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .source-est  { color:#f59e0b; font-size:0.72rem; }
  .preset-box  { background:#0f2744; border-radius:6px; padding:8px 12px; font-size:0.8rem; color:#93c5fd; margin-top:4px; line-height:1.6; }

  .gloss-term { color:#60a5fa; font-weight:700; font-size:0.9rem; margin-top:10px; }
  .gloss-def  { color:#cbd5e1; font-size:0.85rem; margin-left:12px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def fmt_currency(val, prefix="$"):
    if val is None: return "N/A"
    if abs(val) >= 1e12: return f"{prefix}{val/1e12:.2f}T"
    if abs(val) >= 1e9:  return f"{prefix}{val/1e9:.2f}B"
    if abs(val) >= 1e6:  return f"{prefix}{val/1e6:.2f}M"
    return f"{prefix}{val:,.2f}"

def fmt_pct(val, decimals=2):
    if val is None: return "N/A"
    return f"{val*100:.{decimals}f}%"

def fmt_num(val, decimals=2):
    if val is None: return "N/A"
    return f"{val:,.{decimals}f}"

def delta_color(val):
    if val is None: return "#94a3b8"
    return "#22c55e" if val >= 0 else "#ef4444"

def metric_card(label, value, sub=None):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def valuation_card(label, value, current_price, source=None):
    """Renders a fair value card with upside/downside % vs current price."""
    src_html = f'<span class="source-est">{source}</span>' if source else ""
    if value is None:
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid #334155;">
          <div class="metric-label">{label} &nbsp;{src_html}</div>
          <div class="metric-value" style="color:#64748b">N/A</div>
          <div class="metric-sub">Insufficient data</div>
        </div>""", unsafe_allow_html=True)
        return
    val_str = f"${value:,.2f}"
    if current_price:
        diff_pct = (value - current_price) / current_price * 100
        color = "#22c55e" if diff_pct > 5 else ("#ef4444" if diff_pct < -5 else "#eab308")
        arrow = "▲" if diff_pct > 0 else "▼"
        diff_html = f'<span style="color:{color}">{arrow} {abs(diff_pct):.2f}% {"upside" if diff_pct > 0 else "downside"}</span>'
    else:
        diff_html = ""
    st.markdown(f"""
    <div class="metric-card" style="border-left:3px solid #3b82f6;">
      <div class="metric-label">{label} &nbsp;{src_html}</div>
      <div class="metric-value">{val_str}</div>
      <div class="metric-sub">{diff_html}</div>
    </div>""", unsafe_allow_html=True)

def takeaway_box(text):
    st.markdown(f'<div class="takeaway-box"><b>Key Takeaway:</b> {text}</div>',
                unsafe_allow_html=True)

def analyst_rationale_bullets(fund_data, current_price):
    """Generate data-driven bullet points explaining the analyst recommendation."""
    bullets = []
    target = fund_data.get("analyst_target")
    lo     = fund_data.get("analyst_low")
    hi     = fund_data.get("analyst_high")
    count  = fund_data.get("analyst_count")
    if target and current_price:
        upside = (target - current_price) / current_price * 100
        sign   = "+" if upside >= 0 else ""
        cov    = f" ({int(count)} analysts)" if count else ""
        if upside > 15:
            bullets.append(f"Consensus target ${target:.2f}{cov} implies {sign}{upside:.2f}% upside — strong re-rating potential")
        elif upside > 0:
            bullets.append(f"Consensus target ${target:.2f}{cov} implies modest {sign}{upside:.2f}% upside")
        else:
            bullets.append(f"Consensus target ${target:.2f}{cov} implies {sign}{upside:.2f}% downside — limited price upside priced in")
        if lo and hi:
            bullets.append(f"Analyst target range: ${lo:.2f} – ${hi:.2f} (spread of {(hi-lo)/target*100:.0f}% around mean)")

    rev_g = fund_data.get("revenue_growth")
    if rev_g is not None:
        if rev_g > 0.20:
            bullets.append(f"Revenue growing {rev_g*100:.2f}% YoY — high-growth profile supports premium valuation")
        elif rev_g > 0.05:
            bullets.append(f"Revenue growing {rev_g*100:.2f}% YoY — moderate organic expansion")
        elif rev_g >= 0:
            bullets.append(f"Revenue growth is slow at {rev_g*100:.2f}% YoY — watch for acceleration")
        else:
            bullets.append(f"Revenue declining {abs(rev_g)*100:.2f}% YoY — top-line contraction is a red flag")

    earn_g = fund_data.get("earnings_growth")
    if earn_g is not None:
        if earn_g > 0.25:
            bullets.append(f"Earnings growing {earn_g*100:.2f}% — strong EPS expansion can drive re-rating")
        elif earn_g < -0.10:
            bullets.append(f"Earnings contracting {abs(earn_g)*100:.2f}% — margin compression concern")

    margin = fund_data.get("profit_margin")
    if margin is not None:
        if margin > 0.25:
            bullets.append(f"Net margin {margin*100:.2f}% — exceptional profitability and pricing power")
        elif margin > 0.12:
            bullets.append(f"Net margin {margin*100:.2f}% — healthy profitability")
        elif margin < 0:
            bullets.append(f"Net margin negative at {margin*100:.2f}% — company is not yet profitable")

    roe = fund_data.get("roe")
    if roe and roe > 0.15:
        bullets.append(f"ROE of {roe*100:.2f}% indicates efficient use of shareholder capital")

    de = fund_data.get("debt_equity")
    if de is not None:
        if de < 0.3:
            bullets.append(f"Low leverage (D/E {de:.2f}x) — clean balance sheet with financial flexibility")
        elif de > 2.0:
            bullets.append(f"High debt load (D/E {de:.2f}x) — elevated financial risk in rising-rate environment")

    si = fund_data.get("short_pct")
    if si and si > 0.15:
        bullets.append(f"Elevated short interest {si*100:.2f}% — significant institutional skepticism")
    elif si and si < 0.03:
        bullets.append(f"Low short interest {si*100:.2f}% — minimal bearish positioning")

    beta = fund_data.get("beta")
    if beta:
        if beta > 1.5:
            bullets.append(f"High beta ({beta:.2f}) — stock moves more than market; higher risk/reward")
        elif beta < 0.7:
            bullets.append(f"Defensive beta ({beta:.2f}) — lower volatility relative to broad market")

    if not bullets:
        bullets.append("Insufficient quantitative data to generate detailed rationale for this ticker.")
    return bullets


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Stock Analyzer")
    st.markdown("---")

    # Build suggestion index from known universe lists (NASDAQ 100, S&P 100, NIFTY 100)
    # Used for autocomplete only — any ticker can be entered directly regardless.
    _known_stocks = {}
    for _uni in ["NASDAQ 100", "S&P 100", "NIFTY 100"]:
        for _sym, _nm in UNIVERSE_MAP[_uni]:
            if _sym not in _known_stocks:
                _known_stocks[_sym] = _nm

    if "recent_searches" not in st.session_state:
        st.session_state["recent_searches"] = []

    search_query = st.text_input(
        "🔍 Search any US or India stock",
        placeholder="Ticker or company — e.g. AAPL, HDFC, TCS.NS, Tesla...",
        key="stock_search"
    ).strip()

    selected_ticker = None
    selected_name   = None

    if search_query:
        q = search_query.upper()

        # Find autocomplete suggestions from known lists
        if q in _known_stocks:
            _suggestions = [(q, _known_stocks[q])]
        else:
            _prefix  = [(s, n) for s, n in _known_stocks.items() if s.startswith(q)]
            _name_m  = [(s, n) for s, n in _known_stocks.items()
                        if search_query.lower() in n.lower() and (s, n) not in _prefix]
            _suggestions = _prefix + _name_m

        if not _suggestions:
            # Not in suggestion lists — pass directly to Yahoo Finance (any global ticker)
            selected_ticker = q
            selected_name   = q
        elif len(_suggestions) == 1:
            selected_ticker = _suggestions[0][0]
            selected_name   = _suggestions[0][1]
            st.caption(f"✓ {selected_name} ({selected_ticker})")
        else:
            # Show top 20 matches + always allow the raw query as direct entry
            _opts = [f"{s} — {n}" for s, n in _suggestions[:20]]
            _opts.append(f"{q} — (enter directly)")
            _pick = st.selectbox("Select from suggestions", _opts, key="stock_picker")
            if _pick.endswith("(enter directly)"):
                selected_ticker = q
                selected_name   = q
            else:
                _idx = _opts.index(_pick)
                selected_ticker = _suggestions[_idx][0]
                selected_name   = _suggestions[_idx][1]

        # Track recent searches
        if selected_ticker and selected_ticker not in st.session_state["recent_searches"]:
            st.session_state["recent_searches"].insert(0, selected_ticker)
            st.session_state["recent_searches"] = st.session_state["recent_searches"][:5]

    elif st.session_state.get("recent_searches"):
        # No query typed — offer recent searches
        _recent_pick = st.selectbox(
            "Recent searches", st.session_state["recent_searches"], key="recent_picker"
        )
        selected_ticker = _recent_pick
        selected_name   = _known_stocks.get(_recent_pick, _recent_pick)
    else:
        # First ever load — default to AAPL
        selected_ticker = "AAPL"
        selected_name   = "Apple Inc."

    st.markdown("---")
    period = st.radio("Chart Period", ["1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"],
                      index=4, horizontal=True)

    # ── Risk Appetite ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Risk Appetite")
    risk_choice = st.radio("Profile", list(RISK_PRESETS.keys()), index=0, key="risk_profile")
    preset      = RISK_PRESETS[risk_choice]
    st.caption(preset["desc"])

    if risk_choice != "Custom":
        w          = preset["weights"]
        dcf_growth = preset["dcf_growth"] / 100
        weights    = w
        active_w   = {k: v for k, v in w.items() if v > 0}
        lines      = [f"{WEIGHT_LABELS[k]}: {v}%" for k, v in active_w.items()]
        lines.append(f"DCF Growth Rate: {preset['dcf_growth']}%")
        st.markdown(
            '<div class="preset-box">' + "<br>".join(lines) + "</div>",
            unsafe_allow_html=True
        )
    else:
        dcf_growth_pct = st.slider("DCF Growth Rate — Stage 1 (%)", 0, 30, 12, 1, key="dcf_growth_custom")
        dcf_growth     = dcf_growth_pct / 100
        st.caption("Set 0 to exclude a method. Weights auto-normalise to 100%.")
        w_ev_ebit  = st.slider("EV / Operating Earnings (EV/EBIT)",  0, 100, 15, 5, key="w_ev_ebit")
        w_dcf      = st.slider("DCF — Two-Stage 10yr (CAPM WACC)",   0, 100, 30, 5, key="w_dcf")
        w_pe       = st.slider("Forward P/E × Sector Median",        0, 100, 25, 5, key="w_pe")
        w_ev       = st.slider("EV / EBITDA × Sector Median",        0, 100, 20, 5, key="w_ev")
        w_fcf_yld  = st.slider("FCF Yield (Required Return)",        0, 100, 10, 5, key="w_fcf_yield")
        w_ddm      = st.slider("Dividends (DDM)",                    0, 100,  0, 5, key="w_ddm")
        total_w    = w_ev_ebit + w_dcf + w_pe + w_ev + w_fcf_yld + w_ddm
        if total_w > 0:
            st.caption(f"Total: {total_w}  — auto-normalises to 100%")
        else:
            st.warning("Set at least one weight > 0")
        weights = {"ev_ebit": w_ev_ebit, "dcf": w_dcf,       "pe": w_pe,
                   "ev": w_ev,           "fcf_yield": w_fcf_yld, "ddm": w_ddm}

    st.markdown("---")
    st.markdown("#### 🎯 Confluence Engine")
    uce_trade_type    = st.radio("Trade Type", ["Swing", "Day"], index=0,
                                 key="uce_trade_type", horizontal=True)
    uce_max_drawdown  = st.slider("Max Drawdown per Trade (%)", 1.0, 20.0, 5.0, 0.5,
                                  key="uce_max_drawdown",
                                  help="Maximum % loss you accept before stopping out. "
                                       "Drives position sizing and stop placement.")
    st.caption(
        "Swing = 2.5×ATR stop, 30-day alpha · "
        "Day = 1.5×ATR stop, 10-day alpha (uses daily data only)"
    )

    st.markdown("---")
    if st.button("Analyze", type="primary", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown(
        '<div style="color:#64748b;font-size:0.75rem">Data via Yahoo Finance · 15-min cache · '
        'Not financial advice</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────
if not selected_ticker:
    st.info("Select a stock from the sidebar to begin.")
    st.stop()

with st.spinner(f"Loading {selected_ticker}..."):
    fund_data = extract_fundamentals(selected_ticker)
    hist_df   = fetch_history(selected_ticker, period)
    # SPY required for UCE Relative Strength pillar — fetch quietly, never block on failure
    try:
        spy_df = fetch_history("SPY", period)
    except Exception:
        spy_df = None

# Rate-limit / empty-data guard
_price_ok = fund_data.get("current_price") is not None
_data_ok  = bool(fund_data) and _price_ok

if not _data_ok:
    st.error(
        "**Yahoo Finance returned no data for this ticker.**\n\n"
        "This is almost always a **rate-limit** from Yahoo Finance — not a bug. "
        "Streamlit Community Cloud shares its outbound IP with many apps, and Yahoo Finance "
        "briefly blocks requests when too many arrive at once."
    )
    st.info(
        "**What to do:**\n"
        "1. Wait 30–60 seconds, then click **Analyze** in the sidebar to retry.\n"
        "2. If it fails after 2–3 retries, wait 2–3 minutes and try again.\n"
        "3. Try a different ticker first, then return to this one."
    )
    st.stop()

current_price = fund_data.get("current_price")
name          = fund_data.get("name", selected_name)
sector        = fund_data.get("sector", "default")

sector_mults  = fetch_sector_multiples(sector)

day_change = day_change_pct = None
if hist_df is not None and len(hist_df) >= 2:
    day_change     = hist_df["Close"].iloc[-1] - hist_df["Close"].iloc[-2]
    day_change_pct = day_change / hist_df["Close"].iloc[-2] * 100

price_str = f"${current_price:,.2f}" if current_price else "N/A"
chg_str   = (
    f"{'▲' if day_change >= 0 else '▼'} ${abs(day_change):.2f} ({abs(day_change_pct):.2f}%)"
    if day_change is not None else ""
)
chg_color = delta_color(day_change)

st.markdown(f"""
<div class="header-bar">
  <div>
    <span style="font-size:1.3rem;font-weight:700;color:#f1f5f9">{name}</span>
    <span style="color:#94a3b8;margin-left:10px">({selected_ticker})</span>
    <span style="color:#64748b;font-size:0.85rem;margin-left:10px">{fund_data.get('sector','')}</span>
    <span style="color:#475569;font-size:0.8rem;margin-left:10px">{fund_data.get('industry','')}</span>
  </div>
  <div style="text-align:right">
    <span style="font-size:1.5rem;font-weight:700;color:#f1f5f9">{price_str}</span>
    <span style="color:{chg_color};margin-left:10px;font-size:1rem">{chg_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Fundamental Analysis",
    "Technical Analysis",
    "Investment Decision",
    "Glossary",
])

# ══════════════════════════════════════════════════════════════
# TAB 1: FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    eps             = fund_data.get("eps")
    forward_eps     = fund_data.get("forward_eps")
    bvps            = fund_data.get("book_value")
    fcf             = fund_data.get("free_cash_flow")
    ebit            = fund_data.get("ebit")
    shares          = fund_data.get("shares_outstanding")
    pe              = fund_data.get("trailing_pe")
    ebitda          = fund_data.get("ebitda")
    net_debt        = fund_data.get("net_debt", 0)
    div_rate        = fund_data.get("dividend_rate")
    earnings_growth = fund_data.get("earnings_growth")
    peg             = fund_data.get("peg_ratio")
    beta            = fund_data.get("beta") or 1.0

    # ── COMPUTE FAIR VALUES — INSTITUTIONAL GRADE ────────────
    # Two-stage 10yr DCF with CAPM-derived WACC (risk_free=4.5%, ERP=5.5%)
    dcf = dcf_valuation(fcf, dcf_growth, shares, beta=beta)

    # Forward P/E first, trailing fallback
    pe_val, pe_eps_label = pe_based_valuation(eps, forward_eps, sector_mults["pe"])

    # EV/EBITDA — primary institutional multiple
    ev_val = ev_ebitda_valuation(ebitda, net_debt, shares, sector_mults["ev_ebitda"])

    # EV/EBIT — more conservative, strips D&A; PE/HF standard for LBO analysis
    ev_ebit_val = ev_ebit_valuation(ebit, net_debt, shares, sector_mults["ev_ebit"])

    # FCF Yield — anchors price to required investor return threshold
    fcf_yield_val = fcf_yield_valuation(fcf, shares,
                                        required_fcf_yield=sector_mults["fcf_yield_req"])

    # DDM — dividend-paying stocks only
    ddm_growth = 0.04
    ddm        = dividend_discount_model(div_rate, ddm_growth)

    # PEG signal (informational, not a valuation method)
    peg_v, peg_sig = peg_signal(pe, earnings_growth)

    composite = composite_fair_value(
        {"ev_ebit": ev_ebit_val, "dcf": dcf,       "pe": pe_val,
         "ev": ev_val,           "fcf_yield": fcf_yield_val, "ddm": ddm},
        weights
    )
    signal_text, upside_pct = fundamental_signal(current_price, composite)

    metrics_for_score = {
        "roe":           fund_data.get("roe"),
        "net_margin":    fund_data.get("profit_margin"),
        "debt_equity":   fund_data.get("debt_equity"),
        "current_ratio": fund_data.get("current_ratio"),
        "revenue_growth": fund_data.get("revenue_growth"),
        "short_pct":     (fund_data.get("short_pct") or 0) * 100,
    }
    f_score = fundamental_score(current_price, composite, metrics_for_score)

    # ── SIGNAL BANNER ──
    # Green = Undervalued or Fair Value (both positive signals), Red = Overvalued
    banner_color = "#7f1d1d" if "Overvalued" in signal_text else "#14532d"
    border_color = "#ef4444" if "Overvalued" in signal_text else "#22c55e"
    composite_str = f"${composite:,.2f}" if composite else "N/A"
    upside_str    = f"{upside_pct:+.2f}%" if upside_pct is not None else "N/A"

    # Active weight summary
    if risk_choice != "Custom":
        active_wts = {k: v for k, v in weights.items() if v > 0}
        weight_note = " · ".join(f"{WEIGHT_LABELS[k]} {v}%" for k, v in active_wts.items())
        weight_note += f"  |  DCF Growth: {preset['dcf_growth']}%"
    else:
        total_w = sum(weights.values())
        if total_w > 0:
            eff = {k: int(v * 100 / total_w) for k, v in weights.items() if v > 0}
            weight_note = " · ".join(f"{WEIGHT_LABELS[k]} {v}%" for k, v in eff.items())
        else:
            weight_note = "—"

    takeaway_box(
        f"Composite Fair Value is <b>{composite_str}</b> vs current price <b>{price_str}</b> "
        f"({upside_str} {'upside' if upside_pct and upside_pct > 0 else 'downside'}). "
        f"Based on <b>{risk_choice}</b> risk profile. "
        f"Fundamental score: {f_score}%."
    )

    st.markdown(f"""
    <div style="background:{banner_color};border:1px solid {border_color};border-radius:8px;padding:16px 20px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <span style="font-size:1.2rem;font-weight:700;color:#f1f5f9">{signal_text}</span>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">
            Composite Fair Value: <b style="color:#f1f5f9">{composite_str}</b> &nbsp;|&nbsp;
            Current Price: <b style="color:#f1f5f9">{price_str}</b> &nbsp;|&nbsp;
            Upside / Downside: <b style="color:{border_color}">{upside_str}</b>
          </div>
          <div style="color:#64748b;font-size:0.75rem;margin-top:4px">Weights: {weight_note}</div>
        </div>
        <div style="text-align:right">
          <div style="color:#94a3b8;font-size:0.8rem">Fundamental Score</div>
          <div style="font-size:2rem;font-weight:700;color:{border_color}">{f_score}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FAIR VALUE METHODS ──
    wacc_pct = round((0.045 + beta * 0.055) * 100, 1)
    st.markdown(f"### {name} Equity Fair Valuation")
    st.markdown(
        f'<div style="color:#64748b;font-size:0.78rem;margin-bottom:10px">'
        f'CAPM WACC: {wacc_pct}% (β={beta:.2f} · RFR 4.5% · ERP 5.5%) — '
        f'Sector: {sector} · Multiples: Damodaran Jan 2025</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        # Two-stage DCF with CAPM WACC
        valuation_card(
            f"DCF — Two-Stage 10yr ({dcf_growth*100:.0f}% → 3% growth)",
            dcf, current_price,
            f"est. CAPM WACC {wacc_pct}% · 5yr high-growth + 5yr fade"
        )
    with c2:
        pe_lbl = "Forward P/E" if pe_eps_label == "fwd EPS" else "Price to Earnings"
        if pe_eps_label:
            pe_lbl += f" ({pe_eps_label})"
        valuation_card(pe_lbl, pe_val, current_price, sector_mults["pe_source"])
    with c3:
        valuation_card("EV / EBITDA", ev_val, current_price,
                       sector_mults["ev_ebitda_source"])

    c4, c5, c6 = st.columns(3)
    with c4:
        valuation_card("EV / Operating Earnings (EBIT)", ev_ebit_val, current_price,
                       sector_mults["ev_ebit_source"])
    with c5:
        req_yield_pct = round(sector_mults["fcf_yield_req"] * 100, 1)
        valuation_card(
            f"FCF Yield ({req_yield_pct}% required return)",
            fcf_yield_val, current_price,
            sector_mults["fcf_yield_source"]
        )
    with c6:
        if div_rate and div_rate > 0:
            valuation_card("Dividends — DDM (4% growth)", ddm, current_price,
                           "est. (fixed 4% growth)")
        else:
            st.markdown("""
            <div class="metric-card" style="border-left:3px solid #334155">
              <div class="metric-label">Dividends (DDM)</div>
              <div class="metric-value" style="color:#64748b">N/A</div>
              <div class="metric-sub">Not a dividend-paying stock</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#64748b;font-size:0.75rem;margin-bottom:12px">
      est. = sector-consensus median · Damodaran Jan 2025 / Bloomberg aggregates ·
      FCF Yield conservative for high-capex companies (uses trailing FCF)
    </div>""", unsafe_allow_html=True)

    # PEG
    if peg_v is not None:
        peg_color  = "#22c55e" if peg_v < 1 else ("#ef4444" if peg_v > 1.5 else "#eab308")
        own_pe_str = f" (stock P/E: {pe:.2f}x)" if pe else ""
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {peg_color}">
          <div class="metric-label">PEG Ratio{own_pe_str}</div>
          <div class="metric-value" style="color:{peg_color}">{peg_v:.2f} — {peg_sig}</div>
          <div class="metric-sub">
            PEG = P/E divided by Earnings Growth Rate.
            Below 1 = growth justifies valuation. Above 1.5 = expensive relative to growth.<br>
            <span style="color:#64748b;font-size:0.72rem">
              PEG and Composite Fair Value can disagree: PEG rewards fast-growing companies;
              absolute methods (Safe Value, DCF) may still flag them as expensive — both views are valid.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── KEY METRICS ──
    st.markdown("### Key Fundamental Metrics")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Growth**")
        metric_card("Revenue Growth (YoY)",  fmt_pct(fund_data.get("revenue_growth")))
        metric_card("Revenue 3Y CAGR",       fmt_pct(fund_data.get("revenue_3yr_cagr")))
        metric_card("Earnings Growth (YoY)", fmt_pct(fund_data.get("earnings_growth")))
        metric_card("Earnings 3Y CAGR",      fmt_pct(fund_data.get("earnings_3yr_cagr")))
        metric_card("Free Cash Flow",        fmt_currency(fcf),
                    f"Yield: {fmt_pct(fund_data.get('fcf_yield'))}")

    with col_b:
        st.markdown("**Profitability and Quality**")
        metric_card("EBITDA Margin",          fmt_pct(fund_data.get("ebitda_margin")))
        metric_card("Net Profit Margin",      fmt_pct(fund_data.get("profit_margin")))
        metric_card("Return on Equity (ROE)", fmt_pct(fund_data.get("roe")))
        metric_card("Return on Assets (ROA)", fmt_pct(fund_data.get("roa")))
        metric_card("ROCE (Approx.)",         fmt_pct(fund_data.get("roce")))

    with col_c:
        st.markdown("**Risk and Ownership**")
        metric_card("Debt / Equity",      fmt_num(fund_data.get("debt_equity")))
        metric_card("Current Ratio",      fmt_num(fund_data.get("current_ratio")))
        metric_card("Interest Coverage",  fmt_num(fund_data.get("interest_coverage")))
        metric_card("Insider Ownership",  fmt_pct(fund_data.get("insider_ownership")))
        metric_card("Institutional Own.", fmt_pct(fund_data.get("institutional_ownership")))

    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**Analyst Consensus**")

        rec_raw   = fund_data.get("recommendation", "") or ""
        rec       = rec_raw.upper().replace("_", " ").strip() or "N/A"
        rec_mean  = fund_data.get("recommendation_mean")
        ana_count = fund_data.get("analyst_count")

        rec_colors = {
            "BUY": "#15803d", "STRONG BUY": "#14532d", "STRONGBUY": "#14532d",
            "HOLD": "#713f12", "NEUTRAL": "#713f12",
            "SELL": "#7f1d1d", "UNDERPERFORM": "#7f1d1d", "STRONG SELL": "#7f1d1d",
            "OVERWEIGHT": "#15803d", "UNDERWEIGHT": "#7f1d1d",
            "N/A": "#334155",
        }
        rec_color = rec_colors.get(rec.replace(" ", "").upper(),
                    rec_colors.get(rec, "#334155"))

        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Analyst Recommendation</div>
          <div class="metric-value" style="color:{rec_color}">{rec}</div>
          <div class="metric-sub">
            Avg Target: {fmt_currency(fund_data.get('analyst_target'))} &nbsp;|&nbsp;
            Low: {fmt_currency(fund_data.get('analyst_low'))} &nbsp;|&nbsp;
            High: {fmt_currency(fund_data.get('analyst_high'))}
          </div>
        </div>""", unsafe_allow_html=True)

        if rec_mean is not None:
            pct       = max(0, min(100, (rec_mean - 1) / 4 * 100))
            bar_color = "#22c55e" if pct < 30 else ("#ef4444" if pct > 70 else "#eab308")
            cnt_str   = f" · {int(ana_count)} analysts" if ana_count else ""
            st.markdown(f"""
            <div style="margin:-6px 0 10px 0;padding:0 16px">
              <div style="display:flex;justify-content:space-between;color:#64748b;font-size:0.7rem">
                <span>Strong Buy (1)</span><span>Hold (3)</span><span>Strong Sell (5)</span>
              </div>
              <div style="background:#334155;border-radius:4px;height:6px;overflow:hidden;margin:2px 0">
                <div style="background:{bar_color};width:{pct:.0f}%;height:6px;border-radius:4px"></div>
              </div>
              <div style="color:#94a3b8;font-size:0.75rem">Mean score: {rec_mean:.2f} / 5.0{cnt_str}</div>
            </div>""", unsafe_allow_html=True)

        metric_card("Beta (Market Risk)", fmt_num(fund_data.get("beta")),
                    "Beta > 1 = more volatile than market")

        if rec != "N/A":
            with st.expander("Why this recommendation? (Data-Driven Rationale)", expanded=False):
                st.markdown(
                    '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:8px">'
                    'Auto-generated from quantitative signals. Not actual analyst notes.</div>',
                    unsafe_allow_html=True
                )
                for bullet in analyst_rationale_bullets(fund_data, current_price):
                    st.markdown(f"- {bullet}")

    with col_e:
        st.markdown("**52-Week Range**")
        lo52 = fund_data.get("fifty_two_week_low")
        hi52 = fund_data.get("fifty_two_week_high")
        if lo52 and hi52 and current_price:
            pos     = (current_price - lo52) / (hi52 - lo52) if hi52 != lo52 else 0.5
            pos_pct = round(pos * 100, 2)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">52-Week Range</div>
              <div style="display:flex;justify-content:space-between;color:#f1f5f9;font-size:0.9rem;margin-top:6px">
                <span>${lo52:,.2f}</span>
                <span style="color:#3b82f6;font-weight:600">${current_price:,.2f} ({pos_pct:.2f}%)</span>
                <span>${hi52:,.2f}</span>
              </div>
              <div style="background:#334155;border-radius:4px;height:8px;margin-top:6px;overflow:hidden">
                <div style="background:#3b82f6;width:{pos_pct}%;height:8px;border-radius:4px"></div>
              </div>
              <div class="metric-sub">Position within 52-week range</div>
            </div>""", unsafe_allow_html=True)
        else:
            metric_card("52-Week Range", "N/A")

        if div_rate and div_rate > 0:
            metric_card("Dividend Rate", f"${div_rate:.2f}/yr",
                        f"Yield: {fmt_pct(fund_data.get('dividend_yield'))}")
        metric_card("Market Cap", fmt_currency(fund_data.get("market_cap")))
        metric_card("Short Interest %", fmt_pct(fund_data.get("short_pct")),
                    "High > 15% is bearish pressure")


# ══════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if hist_df is None or hist_df.empty:
        st.error(f"No historical data available for {selected_ticker}.")
    else:
        t_score, t_signals = technical_score(hist_df)

        sig_text   = "Bullish" if t_score >= 60 else ("Bearish" if t_score < 40 else "Neutral")
        sig_color  = "#14532d" if t_score >= 60 else ("#7f1d1d" if t_score < 40 else "#713f12")
        sig_border = "#22c55e" if t_score >= 60 else ("#ef4444" if t_score < 40 else "#eab308")

        takeaway_box(
            f"Technical signals are <b>{sig_text}</b> with a score of {t_score}%. "
            f"A score above 60 indicates bullish momentum; below 40 is bearish. "
            f"RSI, MACD, Bollinger Bands, and moving averages all contribute."
        )

        st.markdown(f"""
        <div style="background:{sig_color};border:1px solid {sig_border};border-radius:8px;
        padding:12px 20px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{sig_text}</span>
          <span style="color:{sig_border};font-size:1.5rem;font-weight:700">Technical Score: {t_score}%</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("Signal Breakdown", expanded=False):
            cols = st.columns(2)
            items = list(t_signals.items())
            for i, (k, v) in enumerate(items):
                cols[i % 2].markdown(f"**{k}**: {v}")

        with st.expander("What do these technical indicators mean?", expanded=False):
            st.markdown("""
**Price vs SMA200 (200-day Simple Moving Average)**
The single most-watched line by institutional investors. If price is above SMA200 the stock is in a long-term uptrend; below = downtrend. Fund managers often have hard rules not to buy below SMA200. *Contributes ±12 pts to technical score.*

**Price vs SMA50 (50-day Simple Moving Average)**
Short-to-medium trend filter. Price above SMA50 = momentum is positive over the past 2–3 months. Crossing below SMA50 is an early warning sign. *Contributes ±8 pts.*

**EMA9 vs SMA50 (Short-term momentum crossover)**
EMA9 is the 9-day Exponential Moving Average — it reacts faster to recent price moves than SMA50. When EMA9 crosses above SMA50, it signals that short-term momentum has turned positive. *Contributes ±5 pts.*

**RSI — Relative Strength Index (14-day)**
Measures how fast prices are moving on a 0–100 scale.
- **Above 70**: Overbought — stock may be due for a pullback. *−10 pts.*
- **Below 30**: Oversold — stock may bounce. *+10 pts.*
- **40–65**: Healthy neutral-to-bullish momentum zone. *+5 pts.*
- **65–70 or 30–40**: Caution zone. *0 pts.*

**MACD (Moving Average Convergence Divergence)**
Difference between 12-day and 26-day EMA, smoothed by a 9-day signal line.
- MACD line **above** signal line = bullish momentum building. *+8 pts.*
- MACD line **below** signal line = bearish momentum. *−5 pts.*
The histogram shows the gap — wider = stronger momentum.

**Support and Resistance Levels**
Derived from swing highs/lows in the price history.
- **Near support** (within 3%): Price has bounce potential. *+5 pts.*
- **Near resistance** (within 3%): Selling pressure overhead. *−5 pts.*
Support 1 (S1) = nearest floor; Support 2 (S2) = deeper floor. Resistance 1 (R1) / R2 = nearest ceilings.

**Bollinger Band Width**
Bands are set at ±2 standard deviations around SMA20. When the bands squeeze together (low width), volatility is compressed — this often precedes a sharp move in either direction. Watch for a breakout.

**ATR — Average True Range**
Measures the average daily price swing over 14 days. Higher ATR = more volatile stock. Used to size positions and set stop losses: a common rule is stop loss = entry − 1.5× ATR.

**Technical Score thresholds**
- **Above 60**: Bullish — majority of indicators point positive
- **40–60**: Neutral — mixed signals, wait for confirmation
- **Below 40**: Bearish — majority of indicators point negative
            """)

        st.plotly_chart(price_chart(hist_df, selected_ticker, period),
                        use_container_width=True, config={"displayModeBar": True})

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rsi_chart(hist_df), use_container_width=True,
                            config={"displayModeBar": False})
        with col2:
            st.plotly_chart(macd_chart(hist_df), use_container_width=True,
                            config={"displayModeBar": False})

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bb_width_chart(hist_df), use_container_width=True,
                            config={"displayModeBar": False})
        with col4:
            st.plotly_chart(atr_chart(hist_df), use_container_width=True,
                            config={"displayModeBar": False})

        # ══════════════════════════════════════════════════════
        # UNIVERSAL CONFLUENCE ENGINE (UCE)
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("## 🎯 Universal Confluence Engine")
        st.markdown(
            '<div style="color:#94a3b8;font-size:0.84rem;margin-bottom:14px">'
            'Synthesizes 4 technical pillars into a single high-confidence trade signal. '
            f'Trade type: <b>{uce_trade_type}</b> · '
            f'Max drawdown: <b>{uce_max_drawdown:.1f}%</b>'
            '</div>',
            unsafe_allow_html=True,
        )

        _uce = StockDecisionEngine(
            ticker=selected_ticker,
            df=hist_df,
            spy_df=spy_df if 'spy_df' in dir() else None,
            trade_type=uce_trade_type,
            max_drawdown_pct=uce_max_drawdown,
        )
        _uce_result = _uce.analyze()

        if "error" in _uce_result:
            st.warning(f"Confluence Engine: {_uce_result['error']}")
        else:
            conf        = _uce_result["confidence"]
            rec         = _uce_result["recommendation"]
            rec_color   = _uce_result["rec_color"]
            rec_border  = _uce_result["rec_border"]
            pillars     = _uce_result["pillars"]
            p_mom       = pillars["momentum"]
            p_trend     = pillars["trend"]
            p_vol       = pillars["volatility"]
            p_rs        = pillars["rel_strength"]

            # ── Confidence banner ──
            conf_bar_w = int(conf)
            st.markdown(f"""
            <div style="background:{rec_color};border:1px solid {rec_border};
                        border-radius:8px;padding:16px 20px;margin-bottom:16px">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-size:1.25rem;font-weight:700;color:#f1f5f9">{rec}</span>
                  <div style="color:#94a3b8;font-size:0.83rem;margin-top:6px">
                    Entry: <b style="color:#f1f5f9">${_uce_result['entry_price']:,.2f}</b>
                    &nbsp;·&nbsp; Stop: <b style="color:#ef4444">${_uce_result['effective_stop']:,.2f}</b>
                    &nbsp;({_uce_result['stop_pct']:.1f}% below · {_uce_result['stop_driver']})
                    &nbsp;·&nbsp; Target: <b style="color:#22c55e">${_uce_result['target_price']:,.2f}</b>
                    &nbsp;(+{_uce_result['target_pct']:.1f}% · {_uce_result['reward_multiple']}×R)
                    &nbsp;·&nbsp; Risk/share: <b style="color:#f1f5f9">${_uce_result['risk_per_share']:.2f}</b>
                  </div>
                  <div style="color:#64748b;font-size:0.77rem;margin-top:3px">
                    Position sizing: {_uce_result['shares_per_10k']} shares per $10k
                    (stops out at ${_uce_result['capital_at_risk_10k']:.0f} loss = {uce_max_drawdown:.1f}% of $10k)
                  </div>
                </div>
                <div style="text-align:right">
                  <div style="color:#94a3b8;font-size:0.8rem">Confidence Score</div>
                  <div style="font-size:2.2rem;font-weight:700;color:{rec_border}">{conf:.0f}%</div>
                </div>
              </div>
              <div style="margin-top:10px;background:#0f172a;border-radius:4px;height:8px;overflow:hidden">
                <div style="width:{conf_bar_w}%;height:100%;background:{rec_border};
                            border-radius:4px;transition:width 0.5s"></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          color:#475569;font-size:0.7rem;margin-top:2px">
                <span>0 — Avoid</span><span>50 — Speculative</span><span>80+ — High Confidence</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # ── Pillar Analysis Table ──
            st.markdown("### Pillar Analysis")

            def _score_bar(score, max_score):
                pct = int(score / max_score * 100) if max_score > 0 else 0
                color = "#22c55e" if pct >= 70 else ("#eab308" if pct >= 40 else "#ef4444")
                return (
                    f'<div style="background:#0f172a;border-radius:3px;height:6px;'
                    f'width:80px;display:inline-block;vertical-align:middle">'
                    f'<div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div></div>'
                    f' <span style="color:{color};font-size:0.78rem">{score}/{max_score}</span>'
                )

            rows = [
                {
                    "Pillar":     "1 · Momentum",
                    "Indicator":  f"RSI-14 = <b>{p_mom['rsi']}</b>",
                    "Detail":     f"5-day slope: {p_mom['rsi_slope_5d']:+.1f} → {p_mom['slope_signal']}",
                    "Signal":     p_mom["signal"],
                    "Score":      _score_bar(p_mom["score"], p_mom["max_score"]),
                },
                {
                    "Pillar":     "2 · Trend",
                    "Indicator":  (f"MACD {p_trend['macd']:.4f} / Sig {p_trend['signal_line']:.4f}"
                                   if p_trend.get("macd") is not None else "MACD"),
                    "Detail":     (f"Histogram {p_trend['histogram']:+.4f} · "
                                   f"{'Below Zero ✅' if p_trend.get('below_zero') else 'Above Zero'} · "
                                   f"{p_trend.get('histogram_signal', '')}"
                                   if p_trend.get("macd") is not None else "—"),
                    "Signal":     p_trend["signal"],
                    "Score":      _score_bar(p_trend["score"], p_trend["max_score"]),
                },
                {
                    "Pillar":     "3 · Volatility",
                    "Indicator":  f"BB Width = {p_vol['bb_width']:.5f} ({p_vol['bb_pct_rank']:.0f}th pct)",
                    "Detail":     (f"{'🔴 SQUEEZE ACTIVE' if p_vol['squeeze_active'] else 'No squeeze'} · "
                                   f"ATR-14 = {p_vol['atr_14']} · "
                                   f"Smart Stop = ${p_vol['smart_stop']:,.2f} "
                                   f"({p_vol['atr_mult']}×ATR, -{p_vol['smart_stop_pct']:.1f}%)"),
                    "Signal":     p_vol["signal"],
                    "Score":      _score_bar(p_vol["score"], p_vol["max_score"]),
                },
                {
                    "Pillar":     "4 · Rel. Strength",
                    "Indicator":  (
                        f"{selected_ticker} {p_rs['ticker_return']:+.1f}% / "
                        f"SPY {p_rs['spy_return']:+.1f}%"
                        if p_rs.get("spy_return") is not None
                        else f"{selected_ticker} {p_rs.get('ticker_return', 0):+.1f}% (no SPY)"
                    ),
                    "Detail":     (
                        f"Alpha: {p_rs['alpha']:+.1f}% over {p_rs['lookback_days']}d"
                        if p_rs.get("alpha") is not None else f"({p_rs['lookback_days']}d window)"
                    ),
                    "Signal":     p_rs["signal"],
                    "Score":      _score_bar(p_rs["score"], p_rs["max_score"]),
                },
            ]

            # Render as styled HTML table
            table_html = """
            <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
              <thead>
                <tr style="border-bottom:1px solid #334155;color:#94a3b8">
                  <th style="text-align:left;padding:8px 10px;width:12%">Pillar</th>
                  <th style="text-align:left;padding:8px 10px;width:20%">Indicator</th>
                  <th style="text-align:left;padding:8px 10px;width:28%">Detail</th>
                  <th style="text-align:left;padding:8px 10px;width:28%">Signal</th>
                  <th style="text-align:left;padding:8px 10px;width:12%">Score</th>
                </tr>
              </thead>
              <tbody>
            """
            for i, row in enumerate(rows):
                bg = "#1e293b" if i % 2 == 0 else "#0f172a"
                table_html += f"""
                <tr style="background:{bg};border-bottom:1px solid #1e293b">
                  <td style="padding:9px 10px;color:#94a3b8;font-weight:600">{row['Pillar']}</td>
                  <td style="padding:9px 10px;color:#f1f5f9">{row['Indicator']}</td>
                  <td style="padding:9px 10px;color:#94a3b8">{row['Detail']}</td>
                  <td style="padding:9px 10px;color:#f1f5f9">{row['Signal']}</td>
                  <td style="padding:9px 10px">{row['Score']}</td>
                </tr>"""
            table_html += "</tbody></table>"
            # .strip() prevents leading whitespace from triggering markdown code-block rendering
            st.markdown(table_html.strip(), unsafe_allow_html=True)

            # ── Trade Setup Summary ──
            st.markdown("### Trade Setup")
            ts1, ts2, ts3, ts4, ts5 = st.columns(5)
            def _ts_card(label, value, sub="", color="#f1f5f9"):
                return (
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="color:{color}">{value}</div>'
                    f'<div class="metric-sub">{sub}</div></div>'
                )
            with ts1:
                st.markdown(_ts_card("Entry Price",
                    f"${_uce_result['entry_price']:,.2f}",
                    "Current market price"), unsafe_allow_html=True)
            with ts2:
                st.markdown(_ts_card("Stop Loss",
                    f"${_uce_result['effective_stop']:,.2f}",
                    f"-{_uce_result['stop_pct']:.1f}% · {_uce_result['stop_driver']}",
                    color="#ef4444"), unsafe_allow_html=True)
            with ts3:
                st.markdown(_ts_card("Exit Target",
                    f"${_uce_result['target_price']:,.2f}",
                    f"+{_uce_result['target_pct']:.1f}% · {_uce_result['reward_multiple']}×R",
                    color="#22c55e"), unsafe_allow_html=True)
            with ts4:
                st.markdown(_ts_card("Risk per Share",
                    f"${_uce_result['risk_per_share']:.2f}",
                    "Entry minus stop"), unsafe_allow_html=True)
            with ts5:
                st.markdown(_ts_card("Position Size / $10k",
                    f"{_uce_result['shares_per_10k']} shares",
                    f"Max loss ≈ ${_uce_result['capital_at_risk_10k']:.0f} "
                    f"({uce_max_drawdown:.1f}% of $10k)"), unsafe_allow_html=True)

            # ── Score waterfall note ──
            st.markdown(
                f'<div style="color:#64748b;font-size:0.75rem;margin-top:8px">'
                f'Score breakdown: Momentum {p_mom["score"]}/25 · '
                f'Trend {p_trend["score"]}/35 · '
                f'Volatility {p_vol["score"]}/25 · '
                f'Rel. Strength {p_rs["score"]}/15 · '
                f'<b>Total {conf:.0f}/100</b> · '
                f'Thresholds: &gt;80 High-Confidence Buy · 50–80 Speculative · &lt;50 Avoid'
                f'</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
# TAB 3: INVESTMENT DECISION
# ══════════════════════════════════════════════════════════════
with tab3:
    t_score2, _ = technical_score(hist_df) if (hist_df is not None and not hist_df.empty) else (50, {})
    signal_full, action, combined_score = investment_decision(f_score, t_score2)

    action_colors      = {"BUY": "#15803d", "ACCUMULATE": "#166534", "HOLD": "#854d0e",
                          "REDUCE": "#9a3412", "SELL": "#7f1d1d"}
    action_text_colors = {"BUY": "#dcfce7", "ACCUMULATE": "#bbf7d0", "HOLD": "#fef9c3",
                          "REDUCE": "#ffedd5", "SELL": "#fee2e2"}
    bg = action_colors.get(action, "#334155")
    tc = action_text_colors.get(action, "#f1f5f9")

    takeaway_box(
        f"Combined Score is <b>{combined_score}%</b> (Fundamental 60% + Technical 40%). "
        f"Recommendation: <b>{signal_full}</b>. "
        f"Use the tranche plan below to deploy capital in stages rather than all at once."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(f_score, "Fundamental Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_chart(t_score2, "Technical Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_chart(combined_score, "Combined Score"),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:24px;text-align:center;margin:16px 0">
      <div style="font-size:2.5rem;font-weight:800;color:{tc}">{signal_full}</div>
      <div style="color:{tc};opacity:0.8;font-size:1rem;margin-top:4px">
        Combined Score: {combined_score}% &nbsp;|&nbsp; Fundamental 60% + Technical 40%
      </div>
    </div>""", unsafe_allow_html=True)

    # Score explanation
    with st.expander("How are scores calculated?", expanded=False):
        st.markdown("""
**Fundamental Score (0–100%)** — built from:
- Valuation gap: +25 pts if undervalued >15%, +12 if 5–15%; -25 if overvalued >15%, -12 if 5–15%
- ROE: +10 if >20%, +5 if >15%
- Net Margin: +10 if >25%, +5 if >15%
- Debt/Equity: +5 if <0.5x; -10 if >2x
- Current Ratio: +5 if >1.5; -10 if <1
- Revenue Growth: +10 if >20%; +5 if >10%
- Short Interest: -10 if >15%; -5 if >10%

**Technical Score (0–100%)** — built from:
- Price vs SMA50 / SMA200 (trend direction)
- RSI momentum zone (30–70 bullish; outside = caution)
- MACD signal crossover
- Bollinger Band squeeze (volatility regime)
- EMA9 vs SMA50 (short-term momentum)
- 52-week range position

**Combined Score** = Fundamental × 60% + Technical × 40%

Signal thresholds: BUY ≥ 70 · ACCUMULATE 60–70 · HOLD 45–60 · REDUCE 35–45 · SELL < 35
        """)

    if current_price and hist_df is not None and not hist_df.empty:
        resistance, support = find_support_resistance(hist_df)
        plan, stop_loss, s1, s2, r1, r2 = tranche_plan(action, current_price, support, resistance)

        st.markdown("### Tranche Deployment Plan")
        st.markdown(
            '<div style="color:#94a3b8;font-size:0.82rem;margin-bottom:8px">'
            'Capital is split into 3 tranches deployed at <b>progressively lower prices</b> — '
            'each tranche is at least 5–8% apart. This spreads timing risk: if the stock drops '
            'further you buy more at better prices, reducing your average cost.'
            '</div>',
            unsafe_allow_html=True
        )
        if plan:
            plan_df = pd.DataFrame(plan)
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
        else:
            st.info("No tranche plan available for the current signal.")

        st.markdown("### Risk Metrics")
        r1c, r2c, r3c, r4c = st.columns(4)

        downside_to_s2 = ((s2 - current_price) / current_price * 100) if current_price else None
        analyst_target = fund_data.get("analyst_target")
        upside_analyst = ((analyst_target - current_price) / current_price * 100) if analyst_target and current_price else None
        upside_dcf     = ((dcf - current_price) / current_price * 100) if dcf and current_price else None
        best_upside    = upside_analyst if upside_analyst is not None else upside_dcf
        rr_ratio       = (
            round(best_upside / abs(downside_to_s2), 2)
            if best_upside is not None and downside_to_s2 and downside_to_s2 != 0
            else None
        )

        with r1c:
            ds = f"{downside_to_s2:.2f}%" if downside_to_s2 is not None else "N/A"
            metric_card("Downside to S2", ds, f"Strong support at ${s2:.2f}")
        with r2c:
            us = f"{upside_analyst:+.2f}%" if upside_analyst is not None else "N/A"
            metric_card("Upside to Analyst Target", us, fmt_currency(analyst_target))
        with r3c:
            us2 = f"{upside_dcf:+.2f}%" if upside_dcf is not None else "N/A"
            metric_card("Upside to DCF (Two-Stage)", us2, fmt_currency(dcf))
        with r4c:
            rr       = f"{rr_ratio:.2f}x" if rr_ratio is not None else "N/A"
            rr_color = "#22c55e" if (rr_ratio and rr_ratio >= 2) else ("#ef4444" if (rr_ratio and rr_ratio < 1) else "#eab308")
            st.markdown(f"""<div class="metric-card" style="border-left:3px solid {rr_color}">
              <div class="metric-label">Risk / Reward Ratio</div>
              <div class="metric-value" style="color:{rr_color}">{rr}</div>
              <div class="metric-sub">Target: 2x or above</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #ef4444;border-radius:8px;padding:12px 16px;margin-top:8px">
          <span style="color:#94a3b8;font-size:0.85rem">Suggested Stop Loss: </span>
          <span style="color:#ef4444;font-weight:700;font-size:1rem">${stop_loss:.2f}</span>
          <span style="color:#64748b;font-size:0.8rem;margin-left:8px">
            ({((stop_loss - current_price)/current_price*100):.2f}% from current)
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Risk Flags")
    flags = []
    de     = fund_data.get("debt_equity")
    cr     = fund_data.get("current_ratio")
    si     = fund_data.get("short_pct")
    margin = fund_data.get("profit_margin")
    roe_v  = fund_data.get("roe")

    if de and de > 2:           flags.append(("risk", f"High Debt/Equity: {de:.2f}x (above 2 is elevated)"))
    elif de and de < 0.5:       flags.append(("ok",   f"Low Debt/Equity: {de:.2f}x (strong balance sheet)"))
    if cr and cr < 1:           flags.append(("risk", f"Low Current Ratio: {cr:.2f} (liquidity concern)"))
    elif cr and cr > 2:         flags.append(("ok",   f"Strong Current Ratio: {cr:.2f}"))
    if si and si > 0.15:        flags.append(("risk", f"High Short Interest: {si*100:.2f}% (bearish sentiment)"))
    if margin and margin < 0:   flags.append(("risk", f"Negative Net Margin: {margin*100:.2f}%"))
    elif margin and margin > 0.15: flags.append(("ok", f"Strong Net Margin: {margin*100:.2f}%"))
    if roe_v and roe_v < 0:     flags.append(("risk", f"Negative ROE: {roe_v*100:.2f}%"))
    elif roe_v and roe_v > 0.15: flags.append(("ok",  f"Strong ROE: {roe_v*100:.2f}%"))

    if not flags:
        st.markdown('<div class="ok-flag">No major risk flags detected</div>', unsafe_allow_html=True)
    for flag_type, msg in flags:
        st.markdown(f'<div class="{"risk-flag" if flag_type == "risk" else "ok-flag"}">{msg}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: GLOSSARY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## Glossary of Terms")
    st.markdown("All financial terms and indicators used in this app, explained plainly.")

    def gloss(term, definition):
        st.markdown(
            f'<div class="gloss-term">{term}</div>'
            f'<div class="gloss-def">{definition}</div>',
            unsafe_allow_html=True
        )

    # ── VALUATION ──
    st.markdown("### Equity Fair Valuation Methods")
    gloss("DCF — Two-Stage 10yr (Discounted Cash Flow)",
          "Projects Free Cash Flow over 10 years in two stages: Stage 1 uses the chosen growth rate for 5 years; "
          "Stage 2 linearly fades growth from that rate down to 3% (terminal) over the next 5 years. "
          "Discount rate is CAPM-derived: Risk-Free Rate (4.5%, US 10yr) + Beta × Equity Risk Premium (5.5%). "
          "Terminal value added via Gordon Growth Model. Divides total PV by shares outstanding for per-share value. "
          "10-year two-stage model captures far more value for growth compounders than a simple 5-year model. "
          "Key input: DCF growth rate (Stage 1). Aggressive preset uses 20%; Conservative uses 8%.")
    gloss("Forward P/E × Sector Median",
          "Preferred EPS × Sector Benchmark P/E. Uses forward EPS (analyst consensus 12-month estimate) first — "
          "institutional analysts price on forward, not trailing, earnings. Falls back to trailing EPS if "
          "forward is unavailable. Uses the sector's median P/E, not the stock's own P/E "
          "(which would always reproduce the current price — circular math). "
          "Source: Damodaran Jan 2025 sector tables.")
    gloss("EV / EBITDA",
          "Enterprise Value ÷ EBITDA is the primary institutional multiple. Capital-structure neutral — "
          "unaffected by financing decisions. Formula: Fair EV = EBITDA × Sector Median, then subtract Net Debt "
          "and divide by shares. EBITDA adds back D&A, so it's higher than EBIT — giving a higher EV. "
          "Preferred by investment bankers and PE for acquisitions and cross-sector comparisons.")
    gloss("EV / Operating Earnings (EV/EBIT)",
          "More conservative than EV/EBITDA — EBIT (operating income) does NOT add back depreciation & amortisation. "
          "This penalises capital-intensive businesses (heavy machinery, data centres) relative to asset-light ones. "
          "Used by PE firms for LBO analysis and sell-side for cross-sector comparisons where D&A profiles differ. "
          "Formula: Fair EV = EBIT × Sector Median, subtract Net Debt, divide by shares. "
          "Typically gives a lower fair value than EV/EBITDA for the same company.")
    gloss("FCF Yield (Required Return Method)",
          "Anchors the stock's fair value to the return equity investors require. "
          "Formula: Fair Price = FCF per Share ÷ Required FCF Yield. "
          "Required yield = risk-free rate + equity risk premium for the asset class "
          "(3–3.5% for high-growth tech; 4.5–5% for balanced; 5.5–6% for value/income). "
          "This is conservative for high-capex companies (Amazon, Intel) because trailing FCF is suppressed by "
          "growth investment. Balance it against EV/EBITDA and DCF for a complete picture. "
          "Sector-specific required yields sourced from Damodaran Jan 2025.")
    gloss("Dividends (DDM — Gordon Growth Model)",
          "Intrinsic Value = Dividend Per Share ÷ (Discount Rate − Dividend Growth Rate). "
          "Uses a fixed 4% annual dividend growth rate. Applies only to dividend-paying stocks. "
          "N/A for non-dividend payers. Given 5% weight in the Conservative preset only.")
    gloss("Composite Fair Value",
          "Weighted average of all applicable valuation methods. Methods returning N/A (missing data) are excluded "
          "and remaining weights are auto-normalised to 100% so the composite always uses all available data. "
          "Aggressive preset (EV/EBITDA 40% + Forward P/E 35% + DCF 25%) gives the highest composite. "
          "Conservative preset (DCF 35% + EV/EBIT 25% + P/E 20% + FCF Yield 15% + DDM 5%) gives the lowest.")
    gloss("Upside / Downside %",
          "(Composite Fair Value − Current Price) / Current Price × 100. "
          "Positive = stock trading below intrinsic value (upside potential). "
          "Negative = stock trading above intrinsic value (downside risk). "
          "Above +15% = Undervalued  |  -15% to +15% = Fair Value  |  Below -15% = Overvalued.")

    # ── RISK APPETITE ──
    st.markdown("### Risk Appetite Presets")
    gloss("Aggressive",
          "EV/EBITDA 40% + Forward P/E 35% + Two-Stage DCF 25% at 20% Stage 1 growth. "
          "Designed to give the highest composite fair value. Sector EBITDA multiples and "
          "forward earnings dominate. Best for high-growth, pre-dividend tech and platform companies. "
          "EV/EBIT and FCF Yield excluded — they unfairly penalise high-capex growth businesses.")
    gloss("Moderate",
          "EV/EBITDA 35% + Two-Stage DCF 30% + Forward P/E 20% + FCF Yield 15%. DCF growth 12%. "
          "Reflects a typical sell-side analyst blended approach — EBITDA multiples anchor, "
          "DCF provides cash-flow discipline, FCF Yield adds a return-threshold reality check.")
    gloss("Conservative",
          "Two-Stage DCF 35% + EV/EBIT 25% + P/E 20% + FCF Yield 15% + Dividends (DDM) 5%. DCF growth 8%. "
          "EV/EBIT and FCF Yield are the most conservative anchors — they strip D&A and demand a "
          "return threshold respectively. Designed for capital-preservation-first investors.")

    # ── RATIOS ──
    st.markdown("### Key Financial Ratios")
    gloss("P/E Ratio (Trailing)",
          "Price divided by trailing 12-month EPS. How much investors pay per dollar of past earnings. "
          "High P/E may mean overvalued or high growth expectations. Low P/E may mean cheap or declining earnings.")
    gloss("P/E Ratio (Forward)",
          "Price divided by next 12-month estimated EPS (analyst consensus). More forward-looking than trailing P/E. "
          "Used in Price to Earnings valuation when trailing EPS is negative.")
    gloss("PEG Ratio",
          "P/E divided by Earnings Growth Rate (%). Adjusts P/E for growth. "
          "PEG below 1: growth more than justifies valuation. PEG 1–1.5: fairly priced relative to growth. "
          "PEG above 1.5: expensive relative to growth. "
          "Note: PEG can disagree with absolute value methods — PEG rewards high-growth companies while "
          "Safe Value and DCF may still flag them as expensive.")
    gloss("Debt/Equity (D/E)",
          "Total Debt divided by Shareholders' Equity. Measures financial leverage. "
          "Below 0.5: conservative/low risk. 0.5–2: moderate. Above 2: high leverage, elevated risk.")
    gloss("Current Ratio",
          "Current Assets divided by Current Liabilities. Measures short-term liquidity. "
          "Above 1.5: healthy. 1–1.5: adequate. Below 1: may struggle to meet near-term obligations.")
    gloss("Interest Coverage",
          "EBIT divided by Interest Expense. How many times operating profit covers interest payments. "
          "Above 3x: comfortable. Below 1.5x: financial stress risk.")
    gloss("EPS — Earnings Per Share",
          "Net Income divided by Shares Outstanding. Trailing EPS: actual past 12 months. "
          "Forward EPS: analyst estimate for next 12 months.")
    gloss("Book Value Per Share",
          "Total Shareholders' Equity divided by Shares Outstanding. Net asset value per share.")

    # ── PROFITABILITY ──
    st.markdown("### Profitability Metrics")
    gloss("ROE — Return on Equity",
          "Net Income divided by Shareholders' Equity. Above 15%: strong. Above 20%: excellent. "
          "Negative = destroying shareholder value.")
    gloss("ROA — Return on Assets",
          "Net Income divided by Total Assets. More meaningful for asset-heavy industries.")
    gloss("ROCE — Return on Capital Employed",
          "EBIT divided by (Total Assets − Current Liabilities). "
          "A company with ROCE above its cost of capital is creating value.")
    gloss("Net Profit Margin",
          "Net Income divided by Revenue. Above 15%: excellent. Above 8%: healthy. Negative: unprofitable.")
    gloss("EBITDA Margin",
          "EBITDA divided by Revenue. Pre-interest, pre-tax, pre-depreciation operating margin. "
          "Useful for cross-company comparison.")
    gloss("Free Cash Flow (FCF)",
          "Operating Cash Flow minus Capital Expenditures. The cash generated after funding operations and investments.")
    gloss("FCF Yield",
          "FCF divided by Market Cap. Above 5% is generally attractive.")

    # ── GROWTH ──
    st.markdown("### Growth Metrics")
    gloss("Revenue Growth (YoY)",
          "Year-over-year change in total revenue. Above 10%: strong. Negative: contraction.")
    gloss("Revenue 3Y CAGR",
          "Compound Annual Growth Rate of revenue over 3 years. Smooths out year-to-year volatility.")
    gloss("Earnings Growth (YoY)",
          "Year-over-year change in EPS. Earnings growing faster than revenue signals margin expansion.")
    gloss("Earnings 3Y CAGR",
          "3-year compound annual growth in earnings. A consistent 15%+ CAGR is what most growth investors target.")

    # ── TECHNICAL ──
    st.markdown("### Technical Indicators")
    gloss("SMA — Simple Moving Average",
          "Average closing price over N periods. SMA50 = last 50 days. SMA200 = last 200 days. "
          "Price above SMA200 = long-term uptrend.")
    gloss("EMA — Exponential Moving Average",
          "Like SMA but gives more weight to recent prices, so it reacts faster.")
    gloss("RSI — Relative Strength Index",
          "Momentum oscillator scaled 0–100. Above 70: overbought. Below 30: oversold. 40–65: neutral-to-bullish.")
    gloss("MACD — Moving Average Convergence Divergence",
          "Difference between 12-day EMA and 26-day EMA. Signal line is 9-day EMA of MACD. "
          "MACD crossing above signal = bullish. Below signal = bearish.")
    gloss("Bollinger Bands",
          "SMA20 plus/minus 2 standard deviations. BB Width = (Upper − Lower) / Middle. "
          "Narrow band (squeeze) often precedes a big price move.")
    gloss("ATR — Average True Range",
          "Average of the daily high-low range over 14 periods. Measures volatility in dollar terms. "
          "Useful for setting stop-loss levels: typically 1.5–2x ATR below entry.")
    gloss("Support Level",
          "Price level where buying demand has historically stopped a decline. "
          "S1 = nearest support; S2 = secondary (deeper) support.")
    gloss("Resistance Level",
          "Price level where selling pressure has historically capped a rally. "
          "R1 = nearest resistance; R2 = next higher resistance.")

    # ── MARKET / RISK ──
    st.markdown("### Market and Risk Metrics")
    gloss("Market Capitalisation",
          "Share Price times Shares Outstanding. Large-cap above $10B. Mid-cap $2B–$10B. Small-cap below $2B.")
    gloss("Beta",
          "Sensitivity of the stock to market movements. 1.0 = moves with market. "
          "1.5 = 50% more volatile. 0.5 = half as volatile.")
    gloss("Short Interest %",
          "Shares sold short as % of float. Above 10–15% = many investors betting the stock will fall. "
          "Can also trigger a short squeeze if the stock rises sharply.")
    gloss("Analyst Recommendation Mean",
          "Average of all analyst ratings: 1 = Strong Buy, 2 = Buy, 3 = Hold, 4 = Underperform, 5 = Sell. "
          "A mean below 2.0 indicates strong bullish consensus.")

    # ── INVESTMENT DECISION ──
    st.markdown("### Investment Decision Terms")
    gloss("Fundamental Score (0–100)",
          "Composite score based on: valuation gap to fair value, ROE, net margin, debt/equity, "
          "current ratio, revenue growth, and short interest. Above 70 = bullish fundamentals.")
    gloss("Technical Score (0–100)",
          "Composite score based on: price relative to SMA50/SMA200, RSI momentum, MACD signal, "
          "Bollinger Band squeeze, EMA vs SMA crossover, and 52-week range position.")
    gloss("Combined Score",
          "Fundamental Score × 60% + Technical Score × 40%. "
          "BUY above 70 · ACCUMULATE 60–70 · HOLD 45–60 · REDUCE 35–45 · SELL below 35.")
    gloss("Tranche Plan",
          "Splits capital into 3 tranches deployed at progressively lower prices (for buys) or higher (for sells), "
          "reducing timing risk. Entry points anchored to support levels; exit targets anchored to resistance or analyst targets.")
    gloss("Risk / Reward Ratio",
          "Potential upside (to analyst target or DCF fair value) divided by potential downside (to S2 support). "
          "Target: 2x or above — for every 1% you risk, you should aim for 2% upside.")
    gloss("Stop Loss",
          "A pre-defined exit price to cap downside. Set near S2 support (strong support level). "
          "Typically shown as a % below entry. Helps enforce discipline and protect capital.")
