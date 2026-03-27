"""
Stock Analyzer — Streamlit App
Covers NASDAQ 100, S&P 100, NIFTY 100
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
    ev_ebitda_valuation, pb_valuation, peg_signal,
    dividend_discount_model, composite_fair_value,
    fundamental_signal, fundamental_score,
    technical_score, investment_decision, tranche_plan,
    find_support_resistance,
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
# RISK APPETITE PRESETS
# Aggressive  → highest composite fair value (PE + EV dominate, 20% DCF growth)
# Moderate    → balanced (DCF-led, moderate growth)
# Conservative→ lowest composite fair value (Graham anchors, slow-growth DCF)
# ──────────────────────────────────────────────────────────────
RISK_PRESETS = {
    "Aggressive": {
        "desc": (
            "Growth-focused. Enterprise Market Value and Price to Earnings each take 45%. "
            "Future Cash Flow Value at 10% with 20% growth assumption. "
            "Produces the highest composite fair value — suited for high-growth stocks."
        ),
        "weights":    {"graham": 0,  "dcf": 10, "pe": 45, "ev": 45, "pb": 0,  "ddm": 0},
        "dcf_growth": 20,
    },
    "Moderate": {
        "desc": (
            "Balanced. Future Cash Flow Value leads at 40%, supported by Price to Earnings "
            "and Enterprise Market Value. Small Safe Value buffer. 12% DCF growth rate."
        ),
        "weights":    {"graham": 10, "dcf": 40, "pe": 25, "ev": 20, "pb": 5,  "ddm": 0},
        "dcf_growth": 12,
    },
    "Conservative": {
        "desc": (
            "Safety-first. Safe Value and Future Cash Flow Value anchor the composite. "
            "All six methods active. Lowest composite fair value by design. 8% DCF growth rate."
        ),
        "weights":    {"graham": 20, "dcf": 35, "pe": 20, "ev": 15, "pb": 5,  "ddm": 5},
        "dcf_growth": 8,
    },
    "Custom": {
        "desc": "Set your own weights and growth rate with the controls below.",
        "weights":    None,
        "dcf_growth": None,
    },
}

# Universe browser always uses Aggressive weights (gives broadest upside signal)
UNIVERSE_WEIGHTS    = RISK_PRESETS["Aggressive"]["weights"]
UNIVERSE_DCF_GROWTH = RISK_PRESETS["Aggressive"]["dcf_growth"] / 100

# Human-readable weight key names
WEIGHT_LABELS = {
    "graham": "Safe Value",
    "dcf":    "Future Cash Flow",
    "pe":     "Price to Earnings",
    "ev":     "Enterprise Market Value",
    "pb":     "Book Value",
    "ddm":    "Dividends",
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

    universe = st.selectbox(
        "Stock Universe",
        ["Custom Ticker", "NASDAQ 100", "S&P 100", "NIFTY 100"],
        key="universe"
    )

    if universe == "Custom Ticker":
        ticker_input    = st.text_input("Enter Ticker", value="AAPL", key="custom_ticker").upper().strip()
        selected_ticker = ticker_input
        selected_name   = ticker_input
    else:
        tickers_list    = UNIVERSE_MAP[universe]
        options         = [f"{sym} — {name}" for sym, name in tickers_list]
        selected_option = st.selectbox("Select Stock", options, key="stock_picker")
        idx             = options.index(selected_option)
        selected_ticker = tickers_list[idx][0]
        selected_name   = tickers_list[idx][1]

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
        dcf_growth_pct = st.slider("DCF Growth Rate (%)", 0, 30, 10, 1, key="dcf_growth_custom")
        dcf_growth     = dcf_growth_pct / 100
        st.caption("Set 0 to exclude a method. Weights auto-normalise to 100%.")
        w_graham = st.slider("Safe Value (Graham)",         0, 100, 15, 5, key="w_graham")
        w_dcf    = st.slider("Future Cash Flow (DCF)",      0, 100, 30, 5, key="w_dcf")
        w_pe     = st.slider("Price to Earnings",           0, 100, 25, 5, key="w_pe")
        w_ev     = st.slider("Enterprise Market Value",     0, 100, 15, 5, key="w_ev")
        w_pb     = st.slider("Book Value",                  0, 100, 10, 5, key="w_pb")
        w_ddm    = st.slider("Dividends (DDM)",             0, 100,  5, 5, key="w_ddm")
        total_w  = w_graham + w_dcf + w_pe + w_ev + w_pb + w_ddm
        if total_w > 0:
            st.caption(f"Total: {total_w}  — auto-normalises to 100%")
        else:
            st.warning("Set at least one weight > 0")
        weights = {"graham": w_graham, "dcf": w_dcf, "pe": w_pe,
                   "ev": w_ev,         "pb": w_pb,   "ddm": w_ddm}

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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Fundamental Analysis",
    "Technical Analysis",
    "Investment Decision",
    "Universe Browser",
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
    shares          = fund_data.get("shares_outstanding")
    pe              = fund_data.get("trailing_pe")
    ebitda          = fund_data.get("ebitda")
    net_debt        = fund_data.get("net_debt", 0)
    div_rate        = fund_data.get("dividend_rate")
    earnings_growth = fund_data.get("earnings_growth")
    peg             = fund_data.get("peg_ratio")

    # ── COMPUTE FAIR VALUES ──────────────────────────────────
    graham = graham_number(eps, bvps)
    dcf    = dcf_valuation(fcf, dcf_growth, shares)

    pe_val, pe_eps_label = pe_based_valuation(eps, forward_eps, sector_mults["pe"])
    ev_val = ev_ebitda_valuation(ebitda, net_debt, shares, sector_mults["ev_ebitda"])
    pb_val = pb_valuation(bvps, sector_mults["pb"])

    ddm_growth = 0.04
    ddm        = dividend_discount_model(div_rate, ddm_growth)

    peg_v, peg_sig = peg_signal(pe, earnings_growth)

    composite = composite_fair_value(
        {"graham": graham, "dcf": dcf, "pe": pe_val,
         "ev": ev_val, "pb": pb_val, "ddm": ddm},
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
    est_src = "est. (Damodaran/Bloomberg sector median)"
    st.markdown(f"### {name} Equity Fair Valuation")

    c1, c2, c3 = st.columns(3)
    with c1:
        valuation_card("Safe Value (Graham)", graham, current_price,
                       "est. (formula-based)")
    with c2:
        valuation_card(f"Future Cash Flow Value ({dcf_growth*100:.0f}% growth)", dcf, current_price,
                       "est. (user growth rate)")
    with c3:
        pe_lbl = "Price to Earnings"
        if pe_eps_label:
            pe_lbl += f" ({pe_eps_label})"
        valuation_card(pe_lbl, pe_val, current_price, sector_mults["pe_source"])

    c4, c5, c6 = st.columns(3)
    with c4:
        valuation_card("Enterprise Market Value", ev_val, current_price,
                       sector_mults["ev_ebitda_source"])
    with c5:
        valuation_card("Book Value", pb_val, current_price,
                       sector_mults["pb_source"])
    with c6:
        if div_rate and div_rate > 0:
            valuation_card("Dividends (4% growth est.)", ddm, current_price,
                           "est. (fixed 4% growth)")
        else:
            st.markdown("""
            <div class="metric-card" style="border-left:3px solid #334155">
              <div class="metric-label">Dividends</div>
              <div class="metric-value" style="color:#64748b">N/A</div>
              <div class="metric-sub">Not a dividend-paying stock</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="color:#64748b;font-size:0.75rem;margin-bottom:12px">
      est. = industry-consensus estimate based on Damodaran / Bloomberg sector medians
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
            metric_card("Upside to Future Cash Flow Value", us2, fmt_currency(dcf))
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
# TAB 4: UNIVERSE BROWSER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Stock Universe Browser")

    takeaway_box(
        "Select individual stocks from any index universe to compare. "
        "Fair values use Aggressive weights (Enterprise Market Value 45%, Price to Earnings 45%, "
        "Future Cash Flow Value 10% at 20% growth). "
        "Load in batches of up to 20 stocks at a time to stay within Yahoo Finance rate limits. "
        "Results are cached — switch tabs and come back without re-fetching."
    )

    uni_choice = st.selectbox(
        "Select Universe",
        ["NASDAQ 100", "S&P 100", "NIFTY 100"],
        key="universe_browse"
    )
    all_tickers = UNIVERSE_MAP[uni_choice]   # list of (sym, name)

    all_options = [f"{sym} — {nm}" for sym, nm in all_tickers]

    # Default: first 30 stocks pre-selected
    default_sel = all_options[:30]

    selected_stocks = st.multiselect(
        f"Select stocks to load (max 100, {len(all_tickers)} available in {uni_choice})",
        options=all_options,
        default=default_sel,
        key="uni_multiselect",
        help="Choose individual stocks. Loading 20+ at a time may take 1–2 minutes due to Yahoo Finance rate limits."
    )

    if len(selected_stocks) > 100:
        st.warning("Maximum 100 stocks at a time. Only the first 100 will be loaded.")
        selected_stocks = selected_stocks[:100]

    # Symbol → name lookup
    sym_name_map = {sym: nm for sym, nm in all_tickers}

    col_load, col_clear = st.columns([2, 1])
    load_btn  = col_load.button("Load Selected Stocks", type="primary")
    clear_btn = col_clear.button("Clear Results")

    if clear_btn:
        st.session_state.pop("uni_results_df", None)
        st.session_state.pop("uni_results_key", None)

    # Cache key: sorted list of selected symbols + universe name
    sel_symbols  = [s.split(" — ")[0] for s in selected_stocks]
    cache_key    = uni_choice + "|" + ",".join(sorted(sel_symbols))

    # Load if button pressed or if cache key matches
    if load_btn:
        st.session_state.pop("uni_results_df", None)   # force refresh

    need_load = (
        load_btn or
        ("uni_results_df" not in st.session_state) or
        (st.session_state.get("uni_results_key") != cache_key)
    )

    if selected_stocks and need_load:
        from utils.data import _ticker as _yf_ticker   # shared browser-session Ticker
        rows     = []
        n        = len(sel_symbols)
        progress = st.progress(0)
        status   = st.empty()

        HARDCODED_SECTOR_PE = {
            "Technology": 28, "Consumer Cyclical": 22, "Financial Services": 14,
            "Healthcare": 20, "Communication Services": 22, "Industrials": 18,
            "Consumer Defensive": 18, "Energy": 12, "Utilities": 15,
            "Basic Materials": 14, "Real Estate": 20, "default": 18,
        }
        HARDCODED_EV_EBITDA = {
            "Technology": 20, "Consumer Cyclical": 14, "Financial Services": 12,
            "Healthcare": 15, "Communication Services": 16, "Industrials": 13,
            "Consumer Defensive": 12, "Energy": 7, "Utilities": 10,
            "Basic Materials": 9, "Real Estate": 18, "default": 13,
        }
        HARDCODED_PB = {
            "Technology": 6, "Consumer Cyclical": 4, "Financial Services": 1.4,
            "Healthcare": 5, "Communication Services": 4, "Industrials": 3,
            "Consumer Defensive": 3.5, "Energy": 2, "Utilities": 1.5,
            "Basic Materials": 2, "Real Estate": 1.8, "default": 3,
        }

        for idx_s, sym in enumerate(sel_symbols):
            status.text(f"Loading {sym} ({idx_s + 1}/{n})...")
            progress.progress((idx_s + 1) / n)

            try:
                # Use browser-session Ticker; retry once with 5s pause on rate-limit
                info = _yf_ticker(sym).info or {}
                MIN_KEYS = 20
                if len(info) < MIN_KEYS:
                    time.sleep(5)
                    info = _yf_ticker(sym).info or {}
                if len(info) < MIN_KEYS:
                    # Still rate-limited after retry — try fast_info for price only
                    try:
                        fi = _yf_ticker(sym).fast_info
                        rows.append({"Ticker": sym, "Company": sym_name_map.get(sym, sym),
                                     "Price": round(fi.get("lastPrice") or 0, 2), "Note": "Partial (rate limited)"})
                    except Exception:
                        rows.append({"Ticker": sym, "Company": sym_name_map.get(sym, sym),
                                     "Price": None, "Note": "Rate limited"})
                    continue

                px     = info.get("currentPrice") or info.get("regularMarketPrice")
                eps_t  = info.get("trailingEps")
                eps_f  = info.get("forwardEps")
                bv     = info.get("bookValue")
                ebitda_i = info.get("ebitda")
                shares_i = info.get("sharesOutstanding")
                ev_i   = info.get("enterpriseValue")
                fcf_i  = info.get("freeCashflow")
                sec    = info.get("sector", "default")
                net_d  = (info.get("totalDebt") or 0) - (info.get("totalCash") or 0)

                sec_pe  = HARDCODED_SECTOR_PE.get(sec, HARDCODED_SECTOR_PE["default"])
                sec_ev  = HARDCODED_EV_EBITDA.get(sec, HARDCODED_EV_EBITDA["default"])
                sec_pb  = HARDCODED_PB.get(sec, HARDCODED_PB["default"])

                # Compute fair values
                g_val  = graham_number(eps_t, bv)
                d_val  = dcf_valuation(fcf_i, UNIVERSE_DCF_GROWTH, shares_i)
                p_val, _ = pe_based_valuation(eps_t, eps_f, sec_pe)
                e_val  = ev_ebitda_valuation(ebitda_i, net_d, shares_i, sec_ev)
                b_val  = pb_valuation(bv, sec_pb)

                comp   = composite_fair_value(
                    {"graham": g_val, "dcf": d_val, "pe": p_val, "ev": e_val, "pb": b_val, "ddm": None},
                    UNIVERSE_WEIGHTS
                )

                upside = round((comp - px) / px * 100, 2) if comp and px else None

                if upside is not None:
                    if upside > 15:
                        signal = "Undervalued"
                    elif upside < -15:
                        signal = "Overvalued"
                    else:
                        signal = "Fair Value"
                else:
                    signal = "N/A"

                ana_rec  = (info.get("recommendationKey") or "").upper().replace("_", " ")
                ana_tgt  = info.get("targetMeanPrice")
                roe_i    = info.get("returnOnEquity")
                roce_i   = info.get("returnOnAssets")   # Proxy (ROCE needs statements)
                margin_i = info.get("profitMargins")
                rev_g    = info.get("revenueGrowth")

                entry = round(comp * 0.97, 2) if comp else None
                exit_ = round(ana_tgt if ana_tgt else (comp * 1.15 if comp else None), 2) if (ana_tgt or comp) else None

                rows.append({
                    "Ticker":              sym,
                    "Company":             sym_name_map.get(sym, sym),
                    "Price":               round(px, 2) if px else None,
                    "Safe Value":          round(g_val, 2) if g_val else None,
                    "Future Cash Flow":    round(d_val, 2) if d_val else None,
                    "Price to Earnings":   round(p_val, 2) if p_val else None,
                    "Ent. Market Value":   round(e_val, 2) if e_val else None,
                    "Composite Fair Value": round(comp, 2) if comp else None,
                    "% Upside":            upside,
                    "Signal":              signal,
                    "Analyst Rec":         ana_rec or "N/A",
                    "Analyst Target":      round(ana_tgt, 2) if ana_tgt else None,
                    "Entry Point":         entry,
                    "Exit Point":          exit_,
                    "ROE %":               round(roe_i * 100, 2) if roe_i else None,
                    "ROCE % (Approx)":     round(roce_i * 100, 2) if roce_i else None,
                    "Rev Growth %":        round(rev_g * 100, 2) if rev_g else None,
                    "Net Margin %":        round(margin_i * 100, 2) if margin_i else None,
                })

            except Exception as ex:
                rows.append({"Ticker": sym, "Company": sym_name_map.get(sym, sym),
                             "Price": None, "Note": str(ex)[:40]})

            # 1.5s between every stock — prevents burst throttling on shared IP
            if idx_s + 1 < n:
                time.sleep(1.5)

        progress.empty()
        status.empty()

        if rows:
            df_uni = pd.DataFrame(rows)
            st.session_state["uni_results_df"]  = df_uni
            st.session_state["uni_results_key"] = cache_key
        else:
            st.warning("No data returned. Yahoo Finance may be rate-limiting. Try again in 30 seconds.")

    # ── Display results ──
    if "uni_results_df" in st.session_state and st.session_state.get("uni_results_key") == cache_key:
        df_show = st.session_state["uni_results_df"].copy()

        # Summary counts
        if "Signal" in df_show.columns:
            buys  = (df_show["Signal"] == "Undervalued").sum()
            sells = (df_show["Signal"] == "Overvalued").sum()
            holds = (df_show["Signal"] == "Fair Value").sum()
            st.markdown(
                f"**{len(df_show)} stocks loaded** — "
                f"Undervalued: {buys} · Fair Value: {holds} · Overvalued: {sells} "
                f"(Aggressive weights: Enterprise Market Value 45% / Price to Earnings 45% / Future Cash Flow 10%)"
            )

        # Format and colour the table
        # Round all numeric float columns to 2 decimal places before display
        float_cols = df_show.select_dtypes(include="float").columns.tolist()
        for col in float_cols:
            df_show[col] = df_show[col].apply(
                lambda x: round(x, 2) if pd.notna(x) else x
            )

        if "% Upside" in df_show.columns:
            def color_upside(val):
                if pd.isna(val): return ""
                if val > 15:  return "color: #22c55e; font-weight:600"
                if val < -15: return "color: #ef4444; font-weight:600"
                return "color: #eab308"

            # Build format dict for all float cols (2 decimal places)
            fmt = {col: "{:.2f}" for col in float_cols if col != "% Upside"}
            fmt["% Upside"] = "{:.2f}"

            styled = (
                df_show.style
                .applymap(color_upside, subset=["% Upside"])
                .format(fmt, na_rep="N/A")
            )
            st.dataframe(styled, use_container_width=True, height=500)
        else:
            st.dataframe(df_show, use_container_width=True, height=500)

        # CSV download
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name=f"{uni_choice.replace(' ', '_')}_universe.csv",
            mime="text/csv"
        )
    elif not selected_stocks:
        st.info("Select stocks from the multiselect above and click Load Selected Stocks.")


# ══════════════════════════════════════════════════════════════
# TAB 5: GLOSSARY
# ══════════════════════════════════════════════════════════════
with tab5:
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
    gloss("Safe Value (Graham Number)",
          "Intrinsic value formula by Benjamin Graham: square root of (22.5 × EPS × Book Value Per Share). "
          "Combines earnings and assets into a single conservative price target. "
          "Best suited for value stocks; systematically shows growth stocks as overvalued because it ignores future growth. "
          "Given zero weight in the Aggressive preset.")
    gloss("Future Cash Flow Value (DCF — Discounted Cash Flow)",
          "Projects a company's Free Cash Flow over 5 years at the chosen growth rate, then discounts future cash flows "
          "back to today using a 10% discount rate. Adds a terminal value at 3% perpetual growth. "
          "Divides total by shares outstanding to get per-share intrinsic value. "
          "Most sensitive to the growth rate assumption — a higher rate gives a higher fair value. "
          "Aggressive preset uses 20% growth; Conservative uses 8%.")
    gloss("Price to Earnings",
          "EPS × Sector Benchmark P/E Multiple. Uses the sector's median P/E (industry consensus estimate) "
          "instead of the stock's own P/E — using the stock's own P/E would always produce the current price (circular math). "
          "Falls back to Forward EPS when trailing EPS is negative (loss-making companies). "
          "Given 45% weight in the Aggressive preset — tends to produce higher fair values for most growth stocks.")
    gloss("Enterprise Market Value (EV/EBITDA)",
          "Enterprise Value divided by EBITDA is a capital-structure-neutral valuation ratio. "
          "Here: Intrinsic EV = EBITDA × Sector Median Multiple, then subtract Net Debt and divide by shares. "
          "Good for capital-intensive industries. Given 45% weight in the Aggressive preset — "
          "sector multiples are generally high, especially in technology, driving a higher composite fair value.")
    gloss("Book Value (Price-to-Book)",
          "Book Value Per Share × Sector Median P/B Multiple. Book value is total assets minus liabilities. "
          "Most useful for asset-heavy sectors (Financials, Real Estate). Less meaningful for intangible-heavy tech companies.")
    gloss("Dividends (DDM — Gordon Growth Model)",
          "Intrinsic Value = Dividend Per Share / (Discount Rate − Dividend Growth Rate). "
          "Uses a fixed 4% annual dividend growth rate. Applies only to dividend-paying stocks. "
          "N/A for non-dividend payers. Given 5% weight in the Conservative preset only.")
    gloss("Composite Fair Value",
          "Weighted average of all applicable valuation methods. Methods returning N/A are excluded and "
          "remaining weights are auto-normalised to 100%. "
          "Aggressive preset (PE 45% + Enterprise Market Value 45% + DCF 10%) gives the highest composite. "
          "Conservative preset (Safe Value 20% + DCF 35% + PE 20% + Enterprise Market Value 15% + Book Value 5% + Dividends 5%) gives the lowest.")
    gloss("Upside / Downside %",
          "(Composite Fair Value − Current Price) / Current Price × 100. "
          "Positive = stock trading below intrinsic value (upside potential). "
          "Negative = stock trading above intrinsic value (downside risk). "
          "Above +15% = Undervalued  |  -15% to +15% = Fair Value  |  Below -15% = Overvalued.")

    # ── RISK APPETITE ──
    st.markdown("### Risk Appetite Presets")
    gloss("Aggressive",
          "PE 45% + Enterprise Market Value 45% + Future Cash Flow Value 10% at 20% DCF growth rate. "
          "Designed to give the highest composite fair value. Market-based sector multiples dominate. "
          "Best for high-growth, pre-dividend technology or consumer companies. "
          "Safe Value, Book Value, and Dividends have 0% weight.")
    gloss("Moderate",
          "Future Cash Flow Value 40% + Price to Earnings 25% + Enterprise Market Value 20% + "
          "Safe Value 10% + Book Value 5%. DCF growth rate of 12%. "
          "Balanced between cash-flow fundamentals and market multiples.")
    gloss("Conservative",
          "Safe Value 20% + Future Cash Flow Value 35% + Price to Earnings 20% + "
          "Enterprise Market Value 15% + Book Value 5% + Dividends 5%. DCF growth rate of 8%. "
          "All six methods active. Designed to give the lowest composite fair value — "
          "emphasises capital preservation over upside capture.")

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
