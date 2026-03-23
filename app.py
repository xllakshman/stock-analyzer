"""
Stock Analyzer — Streamlit App
Covers NASDAQ 100, S&P 100, NIFTY 100
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np

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
# ──────────────────────────────────────────────────────────────
RISK_PRESETS = {
    "🚀 Aggressive": {
        "desc": "Growth-focused: DCF & Market Value dominate. Graham, Book Value, Dividends excluded. Best for high-growth companies.",
        "weights": {"graham": 0,  "dcf": 50, "pe": 25, "ev": 25, "pb": 0,  "ddm": 0},
    },
    "⚖️ Moderate": {
        "desc": "Balanced: DCF leads with support from PE & EV/EBITDA. Small Graham buffer included.",
        "weights": {"graham": 10, "dcf": 40, "pe": 25, "ev": 20, "pb": 5,  "ddm": 0},
    },
    "🛡️ Conservative": {
        "desc": "Safety-first: All methods active. Graham, Book Value & Dividends anchor the composite. Equal weight on DCF/PE/EV.",
        "weights": {"graham": 10, "dcf": 27, "pe": 27, "ev": 26, "pb": 5,  "ddm": 5},
    },
    "⚙️ Custom": {
        "desc": "Set your own weights with the sliders below.",
        "weights": None,
    },
}

# Universe browser always uses Aggressive preset weights for fair value column
UNIVERSE_WEIGHTS = RISK_PRESETS["🚀 Aggressive"]["weights"]

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

  .header-bar {
    background: #1e293b; border-radius: 8px; padding: 12px 20px;
    margin-bottom: 16px; border-left: 4px solid #3b82f6;
    display: flex; justify-content: space-between; align-items: center;
  }
  .takeaway-box {
    background: #1e3a5f; border: 1px solid #2563eb; border-left: 4px solid #3b82f6;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; font-size: 0.88rem; color: #cbd5e1;
  }

  /* Tab: active = blue text only, no background box */
  .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; border-radius: 6px; font-weight: 500; background: transparent !important; }
  .stTabs [aria-selected="true"] { background: transparent !important; color: #3b82f6 !important; font-weight: 700 !important; border-bottom: 2px solid #3b82f6 !important; }

  .stDataFrame { background: #1e293b !important; }
  .stTextInput input, .stSelectbox select { background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
  .stNumberInput input { background: #1e293b !important; color: #f1f5f9 !important; }
  hr { border-color: #334155; }
  .stProgress > div > div { background-color: #3b82f6 !important; }
  .risk-flag { background:#7f1d1d; color:#fca5a5; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .ok-flag   { background:#14532d; color:#86efac; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .source-est { color:#f59e0b; font-size:0.72rem; }

  /* Glossary */
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
          <div class="metric-label">{label} {src_html}</div>
          <div class="metric-value" style="color:#64748b">N/A</div>
          <div class="metric-sub">Insufficient data</div>
        </div>""", unsafe_allow_html=True)
        return
    val_str = f"${value:,.2f}"
    diff_html = ""
    if current_price:
        diff_pct = (value - current_price) / current_price * 100
        color  = "#22c55e" if diff_pct > 5 else ("#ef4444" if diff_pct < -5 else "#eab308")
        arrow  = "▲" if diff_pct > 0 else "▼"
        diff_html = f'<span style="color:{color}">{arrow} {abs(diff_pct):.1f}% {"upside" if diff_pct>0 else "downside"}</span>'
    st.markdown(f"""
    <div class="metric-card" style="border-left:3px solid #3b82f6;">
      <div class="metric-label">{label} {src_html}</div>
      <div class="metric-value">{val_str}</div>
      <div class="metric-sub">{diff_html}</div>
    </div>""", unsafe_allow_html=True)

def takeaway_box(text):
    st.markdown(f'<div class="takeaway-box">💡 <b>Key Takeaway:</b> {text}</div>',
                unsafe_allow_html=True)

def analyst_rationale_bullets(fund_data, current_price):
    """Generate data-driven bullet points for analyst recommendation rationale."""
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
            bullets.append(f"🟢 Consensus target ${target:.2f}{cov} implies {sign}{upside:.1f}% upside — strong re-rating potential")
        elif upside > 0:
            bullets.append(f"🟡 Consensus target ${target:.2f}{cov} implies modest {sign}{upside:.1f}% upside")
        else:
            bullets.append(f"🔴 Consensus target ${target:.2f}{cov} implies {sign}{upside:.1f}% downside")
        if lo and hi:
            bullets.append(f"📊 Analyst target range: ${lo:.2f} – ${hi:.2f} (spread {(hi-lo)/target*100:.0f}% around mean)")
    rev_g = fund_data.get("revenue_growth")
    if rev_g is not None:
        if rev_g > 0.20:   bullets.append(f"🟢 Revenue growing {rev_g*100:.1f}% YoY — high-growth profile")
        elif rev_g > 0.05: bullets.append(f"🟡 Revenue growing {rev_g*100:.1f}% YoY — moderate expansion")
        elif rev_g >= 0:   bullets.append(f"🟡 Revenue growth slow at {rev_g*100:.1f}% YoY")
        else:              bullets.append(f"🔴 Revenue declining {abs(rev_g)*100:.1f}% YoY — red flag")
    earn_g = fund_data.get("earnings_growth")
    if earn_g is not None:
        if earn_g > 0.25:   bullets.append(f"🟢 Earnings growing {earn_g*100:.1f}% — strong EPS expansion")
        elif earn_g < -0.10: bullets.append(f"🔴 Earnings contracting {abs(earn_g)*100:.1f}% — margin concern")
    margin = fund_data.get("profit_margin")
    if margin is not None:
        if margin > 0.25:   bullets.append(f"🟢 Net margin {margin*100:.1f}% — exceptional profitability")
        elif margin > 0.12: bullets.append(f"🟡 Net margin {margin*100:.1f}% — healthy")
        elif margin < 0:    bullets.append(f"🔴 Net margin negative at {margin*100:.1f}% — unprofitable")
    roe = fund_data.get("roe")
    if roe and roe > 0.15:
        bullets.append(f"🟢 ROE of {roe*100:.1f}% — efficient use of shareholder capital")
    de = fund_data.get("debt_equity")
    if de is not None:
        if de < 0.3:  bullets.append(f"🟢 Low leverage (D/E {de:.2f}x) — financial flexibility")
        elif de > 2:  bullets.append(f"🔴 High debt load (D/E {de:.2f}x) — elevated financial risk")
    si = fund_data.get("short_pct")
    if si and si > 0.15:
        bullets.append(f"🔴 Elevated short interest {si*100:.1f}% — institutional skepticism")
    elif si and si < 0.03:
        bullets.append(f"🟢 Low short interest {si*100:.1f}% — minimal bearish positioning")
    beta = fund_data.get("beta")
    if beta:
        if beta > 1.5:  bullets.append(f"⚠️ High beta ({beta:.2f}) — above-market volatility")
        elif beta < 0.7: bullets.append(f"🟢 Defensive beta ({beta:.2f}) — lower volatility")
    if not bullets:
        bullets.append("ℹ️ Insufficient quantitative data for detailed rationale.")
    return bullets


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Analyzer")
    st.markdown("---")

    universe = st.selectbox("📊 Stock Universe",
        ["Custom Ticker", "NASDAQ 100", "S&P 100", "NIFTY 100"], key="universe")

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
    period = st.radio("Chart Period", ["1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"], index=4, horizontal=True)

    st.markdown("---")
    dcf_growth = st.slider("FCF Growth Rate (%)", 0, 30, 10, 1) / 100

    # ── Risk Appetite & Composite Weights ──
    st.markdown("---")
    st.markdown("#### 🎯 Risk Appetite")
    risk_choice = st.radio("Profile", list(RISK_PRESETS.keys()), index=1, key="risk_profile")
    preset = RISK_PRESETS[risk_choice]
    st.caption(preset["desc"])

    if preset["weights"] is not None:
        weights = preset["weights"]
        w = weights
        st.markdown(
            f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:6px;line-height:1.8">'
            f'Safe Value: <b>{w["graham"]}%</b> &nbsp;·&nbsp; '
            f'FCF Value: <b>{w["dcf"]}%</b> &nbsp;·&nbsp; '
            f'Price/Earnings: <b>{w["pe"]}%</b><br>'
            f'Market Value: <b>{w["ev"]}%</b> &nbsp;·&nbsp; '
            f'Book Value: <b>{w["pb"]}%</b> &nbsp;·&nbsp; '
            f'Dividends: <b>{w["ddm"]}%</b></div>',
            unsafe_allow_html=True
        )
    else:
        w_graham = st.slider("Safe Value (Graham)",   0, 100, 15, 5, key="w_graham")
        w_dcf    = st.slider("FCF Value (DCF)",       0, 100, 30, 5, key="w_dcf")
        w_pe     = st.slider("Price to Earnings",     0, 100, 25, 5, key="w_pe")
        w_ev     = st.slider("Market Value (EV/EBITDA)", 0, 100, 15, 5, key="w_ev")
        w_pb     = st.slider("Book Value (P/B)",      0, 100, 10, 5, key="w_pb")
        w_ddm    = st.slider("Dividends (DDM)",       0, 100,  5, 5, key="w_ddm")
        weights  = {"graham": w_graham, "dcf": w_dcf, "pe": w_pe,
                    "ev": w_ev, "pb": w_pb, "ddm": w_ddm}

    st.markdown("---")
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown(
        '<div style="color:#64748b;font-size:0.75rem">Data via Yahoo Finance · 15-min cache · '
        'Not financial advice</div>', unsafe_allow_html=True)


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
if not fund_data or fund_data.get("current_price") is None:
    st.error("⚠️ **Yahoo Finance returned no data.** This is almost always a temporary rate-limit from their servers.")
    st.info(
        "**What to do:** Wait 30–60 seconds, then click **🔍 Analyze** to retry. "
        "If it persists, try a different ticker first to warm up the connection, then return to this one."
    )
    st.stop()

current_price = fund_data.get("current_price")
name          = fund_data.get("name", selected_name)
sector        = fund_data.get("sector", "default")
sector_mults  = fetch_sector_multiples(sector)

# Day change from history
day_change = day_change_pct = None
if hist_df is not None and len(hist_df) >= 2:
    day_change     = hist_df["Close"].iloc[-1] - hist_df["Close"].iloc[-2]
    day_change_pct = day_change / hist_df["Close"].iloc[-2] * 100

price_str = f"${current_price:,.2f}" if current_price else "N/A"
chg_str   = (f"{'▲' if day_change >= 0 else '▼'} ${abs(day_change):.2f} ({abs(day_change_pct):.2f}%)"
             if day_change is not None else "")
chg_color = delta_color(day_change)

st.markdown(f"""
<div class="header-bar">
  <div>
    <span style="font-size:1.3rem;font-weight:700;color:#f1f5f9">{name}</span>
    <span style="color:#94a3b8;margin-left:10px">({selected_ticker})</span>
    <span style="color:#64748b;font-size:0.85rem;margin-left:10px">{fund_data.get('sector','')}</span>
    <span style="color:#475569;font-size:0.8rem;margin-left:8px">{fund_data.get('industry','')}</span>
  </div>
  <div style="text-align:right">
    <span style="font-size:1.5rem;font-weight:700;color:#f1f5f9">{price_str}</span>
    <span style="color:{chg_color};margin-left:10px;font-size:1rem">{chg_str}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── COMPUTE ALL VALUATIONS (shared across tabs) ──────────────
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

graham_val                  = graham_number(eps, bvps)
dcf_val                     = dcf_valuation(fcf, dcf_growth, shares)
pe_val, pe_eps_label        = pe_based_valuation(eps, forward_eps, sector_mults["pe"])
ev_val                      = ev_ebitda_valuation(ebitda, net_debt, shares, sector_mults["ev_ebitda"])
pb_val                      = pb_valuation(bvps, sector_mults["pb"])
ddm_val                     = dividend_discount_model(div_rate, 0.04)
peg_v, peg_sig              = peg_signal(pe, earnings_growth)

composite = composite_fair_value(
    {"graham": graham_val, "dcf": dcf_val, "pe": pe_val,
     "ev": ev_val, "pb": pb_val, "ddm": ddm_val},
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

# Technical score (shared)
t_score, t_signals = technical_score(hist_df) if (hist_df is not None and not hist_df.empty) else (50, {})

# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Fundamental Analysis",
    "📈 Technical Analysis",
    "🎯 Investment Decision",
    "🏦 Universe Browser",
    "📖 Glossary",
])


# ══════════════════════════════════════════════════════════════
# TAB 1: FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    composite_str = f"${composite:,.2f}" if composite else "N/A"
    upside_str    = f"{upside_pct:+.1f}%" if upside_pct is not None else "N/A"

    # ── Key Takeaway ──
    if composite and current_price:
        _gap = abs(upside_pct)
        _trend = "undervalued" if upside_pct > 15 else ("overvalued" if upside_pct < -15 else "fairly valued")
        _top_metric = []
        if fund_data.get("revenue_growth") and fund_data["revenue_growth"] > 0.1:
            _top_metric.append(f"strong revenue growth ({fund_data['revenue_growth']*100:.0f}% YoY)")
        if fund_data.get("roe") and fund_data["roe"] > 0.15:
            _top_metric.append(f"solid ROE ({fund_data['roe']*100:.0f}%)")
        if fund_data.get("profit_margin") and fund_data["profit_margin"] > 0.1:
            _top_metric.append(f"healthy margins ({fund_data['profit_margin']*100:.0f}%)")
        _strengths = "; ".join(_top_metric) if _top_metric else "see metrics below"
        takeaway_box(
            f"{name} is <b>{_trend}</b> — Composite Fair Value {composite_str} vs current {price_str} "
            f"({upside_str} upside/downside). Risk profile: <b>{risk_choice}</b>. "
            f"Key positives: {_strengths}."
        )

    # ── Signal Banner ──
    banner_color = "#14532d" if "Undervalued" in signal_text else ("#7f1d1d" if "Overvalued" in signal_text else "#713f12")
    border_color = "#22c55e" if "Undervalued" in signal_text else ("#ef4444" if "Overvalued" in signal_text else "#eab308")
    w = weights
    total_w = sum(w.values())
    eff_str = " · ".join(f"{k.upper()}:{int(v*100/total_w)}%" for k, v in w.items() if v > 0) if total_w > 0 else "—"

    st.markdown(f"""
    <div style="background:{banner_color};border:1px solid {border_color};border-radius:8px;padding:16px 20px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <span style="font-size:1.2rem;font-weight:700;color:#f1f5f9">{signal_text}</span>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">
            Composite Fair Value: <b style="color:#f1f5f9">{composite_str}</b> &nbsp;|&nbsp;
            Current: <b style="color:#f1f5f9">{price_str}</b> &nbsp;|&nbsp;
            Upside/Downside: <b style="color:{border_color}">{upside_str}</b>
          </div>
          <div style="color:#64748b;font-size:0.72rem;margin-top:3px">Weights ({risk_choice}): {eff_str}</div>
        </div>
        <div style="text-align:right">
          <div style="color:#94a3b8;font-size:0.8rem">Fundamental Score</div>
          <div style="font-size:2rem;font-weight:700;color:{border_color}">{f_score}/100</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Fair Value Methods — renamed ──
    st.markdown(f"### {name} Equity Fair Valuation")

    est_src = sector_mults.get("pe_source", "⚠️ est.")

    c1, c2, c3 = st.columns(3)
    with c1:
        valuation_card("Safe Value (Graham)", graham_val, current_price, "⚠️ est. (formula-based)")
    with c2:
        valuation_card(f"Future Cash Flow Value (DCF, {dcf_growth*100:.0f}% growth)", dcf_val, current_price, "⚠️ est. (user growth rate)")
    with c3:
        pe_lbl = f"Price to Earnings ({sector_mults['pe']}x"
        if pe_eps_label:
            pe_lbl += f" × {pe_eps_label}"
        pe_lbl += ")"
        valuation_card(pe_lbl, pe_val, current_price, est_src)

    c4, c5, c6 = st.columns(3)
    with c4:
        valuation_card(f"Market Value / EV/EBITDA ({sector_mults['ev_ebitda']}x)", ev_val, current_price, est_src)
    with c5:
        valuation_card(f"Book Value / P/B ({sector_mults['pb']}x)", pb_val, current_price, est_src)
    with c6:
        if div_rate and div_rate > 0:
            valuation_card("Dividends / DDM (4% growth)", ddm_val, current_price, "⚠️ est. (fixed 4% growth)")
        else:
            st.markdown("""
            <div class="metric-card" style="border-left:3px solid #334155">
              <div class="metric-label">Dividends / DDM</div>
              <div class="metric-value" style="color:#64748b">N/A</div>
              <div class="metric-sub">Not a dividend-paying stock</div>
            </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div style="color:#64748b;font-size:0.73rem;margin-bottom:10px">'
        '⚠️ est. = industry-consensus estimate (Damodaran/Bloomberg sector medians). '
        'These are benchmarks, not exact values — treat as indicative ranges.</div>',
        unsafe_allow_html=True)

    # PEG
    if peg_v is not None:
        peg_color  = "#22c55e" if peg_v < 1 else ("#ef4444" if peg_v > 1.5 else "#eab308")
        own_pe_str = f" (stock P/E: {pe:.1f}x)" if pe else ""
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {peg_color}">
          <div class="metric-label">PEG Ratio{own_pe_str}</div>
          <div class="metric-value" style="color:{peg_color}">{peg_v:.2f} — {peg_sig}</div>
          <div class="metric-sub">
            PEG = P/E ÷ Earnings Growth Rate. &lt;1 = growth justifies valuation, &gt;1.5 = expensive relative to growth.<br>
            <span style="color:#64748b;font-size:0.72rem">
              ⚠️ PEG and Composite Fair Value can disagree — PEG rewards growth;
              absolute methods (Graham, DCF) may still flag high-growth stocks as expensive. Both views are valid.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Key Metrics ──
    st.markdown("### Key Fundamental Metrics")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**📈 Growth**")
        metric_card("Revenue Growth (YoY)",  fmt_pct(fund_data.get("revenue_growth")))
        metric_card("Revenue 3Y CAGR",       fmt_pct(fund_data.get("revenue_3yr_cagr")))
        metric_card("Earnings Growth (YoY)", fmt_pct(fund_data.get("earnings_growth")))
        metric_card("Earnings 3Y CAGR",      fmt_pct(fund_data.get("earnings_3yr_cagr")))
        metric_card("Free Cash Flow",         fmt_currency(fcf), f"Yield: {fmt_pct(fund_data.get('fcf_yield'))}")

    with col_b:
        st.markdown("**💰 Profitability & Quality**")
        metric_card("EBITDA Margin",          fmt_pct(fund_data.get("ebitda_margin")))
        metric_card("Net Profit Margin",      fmt_pct(fund_data.get("profit_margin")))
        metric_card("Return on Equity (ROE)", fmt_pct(fund_data.get("roe")))
        metric_card("Return on Assets (ROA)", fmt_pct(fund_data.get("roa")))
        metric_card("ROCE (Approx.)",         fmt_pct(fund_data.get("roce")))

    with col_c:
        st.markdown("**⚖️ Risk & Ownership**")
        metric_card("Debt / Equity",      fmt_num(fund_data.get("debt_equity")))
        metric_card("Current Ratio",      fmt_num(fund_data.get("current_ratio")))
        metric_card("Interest Coverage",  fmt_num(fund_data.get("interest_coverage")))
        metric_card("Insider Ownership",  fmt_pct(fund_data.get("insider_ownership")))
        metric_card("Institutional Own.", fmt_pct(fund_data.get("institutional_ownership")))

    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**🎯 Analyst Consensus**")

        rec_raw  = (fund_data.get("recommendation") or "").upper().replace("_", " ").strip() or "N/A"
        rec_mean = fund_data.get("recommendation_mean")
        ana_cnt  = fund_data.get("analyst_count")

        rec_colors = {
            "BUY": "#15803d", "STRONG BUY": "#14532d", "STRONGBUY": "#14532d",
            "HOLD": "#713f12", "NEUTRAL": "#713f12",
            "SELL": "#7f1d1d", "UNDERPERFORM": "#7f1d1d", "STRONG SELL": "#7f1d1d",
            "OVERWEIGHT": "#15803d", "UNDERWEIGHT": "#7f1d1d", "N/A": "#334155",
        }
        rec_color = rec_colors.get(rec_raw.replace(" ", "").upper(), "#334155")

        # Simple card with no embedded complex HTML
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Analyst Recommendation</div>
          <div class="metric-value" style="color:{rec_color}">{rec_raw}</div>
          <div class="metric-sub">
            Avg Target: {fmt_currency(fund_data.get('analyst_target'))} &nbsp;|&nbsp;
            Low: {fmt_currency(fund_data.get('analyst_low'))} &nbsp;|&nbsp;
            High: {fmt_currency(fund_data.get('analyst_high'))}
          </div>
        </div>""", unsafe_allow_html=True)

        # Mean score bar — rendered as a SEPARATE markdown call to avoid nesting issues
        if rec_mean is not None:
            pct       = max(0, min(100, (rec_mean - 1) / 4 * 100))
            bar_color = "#22c55e" if pct < 30 else ("#ef4444" if pct > 70 else "#eab308")
            cnt_txt   = f" · {int(ana_cnt)} analysts" if ana_cnt else ""
            st.markdown(f"""
            <div style="margin:-6px 0 10px 0;padding:10px 16px;background:#1e293b;border:1px solid #334155;border-radius:0 0 8px 8px">
              <div style="display:flex;justify-content:space-between;color:#64748b;font-size:0.7rem">
                <span>Strong Buy (1)</span><span>Hold (3)</span><span>Strong Sell (5)</span>
              </div>
              <div style="background:#334155;border-radius:4px;height:6px;overflow:hidden;margin:4px 0">
                <div style="background:{bar_color};width:{pct:.0f}%;height:6px;border-radius:4px"></div>
              </div>
              <div style="color:#94a3b8;font-size:0.75rem">Mean: {rec_mean:.1f}/5.0{cnt_txt}</div>
            </div>""", unsafe_allow_html=True)

        metric_card("Beta (Market Risk)", fmt_num(fund_data.get("beta")), "Beta >1 = more volatile than market")

        if rec_raw != "N/A":
            with st.expander("💡 Why this recommendation? (Data-Driven Rationale)", expanded=False):
                st.markdown('<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:8px">Auto-generated from quantitative signals. Not actual analyst notes.</div>',
                            unsafe_allow_html=True)
                for bullet in analyst_rationale_bullets(fund_data, current_price):
                    st.markdown(f"- {bullet}")

    with col_e:
        st.markdown("**📊 Market Data**")
        lo52 = fund_data.get("fifty_two_week_low")
        hi52 = fund_data.get("fifty_two_week_high")
        if lo52 and hi52 and current_price:
            pos     = (current_price - lo52) / (hi52 - lo52) if hi52 != lo52 else 0.5
            pos_pct = round(pos * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">52-Week Range</div>
              <div style="display:flex;justify-content:space-between;color:#f1f5f9;font-size:0.9rem;margin-top:6px">
                <span>${lo52:,.2f}</span>
                <span style="color:#3b82f6;font-weight:600">${current_price:,.2f} ({pos_pct}%)</span>
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
            metric_card("Dividend Rate", f"${div_rate:.2f}/yr", f"Yield: {fmt_pct(fund_data.get('dividend_yield'))}")
        metric_card("Market Cap",        fmt_currency(fund_data.get("market_cap")))
        metric_card("Short Interest %",  fmt_pct(fund_data.get("short_pct")), "High >15% is bearish pressure")


# ══════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if hist_df is None or hist_df.empty:
        st.error(f"No historical data available for {selected_ticker}.")
    else:
        sig_text   = "🟢 Bullish" if t_score >= 60 else ("🔴 Bearish" if t_score < 40 else "🟡 Neutral")
        sig_color  = "#14532d" if t_score >= 60 else ("#7f1d1d" if t_score < 40 else "#713f12")
        sig_border = "#22c55e" if t_score >= 60 else ("#ef4444" if t_score < 40 else "#eab308")

        # Key Takeaway
        _top_signals = [v for v in list(t_signals.values())[:3]]
        takeaway_box(
            f"Technical trend for {name} is <b>{sig_text.replace('🟢','').replace('🔴','').replace('🟡','').strip()}</b> "
            f"with a score of <b>{t_score}/100</b>. "
            + (f"Key signals: {', '.join(_top_signals[:2])}." if _top_signals else "")
        )

        st.markdown(f"""
        <div style="background:{sig_color};border:1px solid {sig_border};border-radius:8px;
        padding:12px 20px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{sig_text}</span>
          <span style="color:{sig_border};font-size:1.5rem;font-weight:700">Technical Score: {t_score}/100</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 Signal Breakdown", expanded=False):
            cols = st.columns(2)
            for i, (k, v) in enumerate(t_signals.items()):
                cols[i % 2].markdown(f"**{k}**: {v}")

        st.plotly_chart(price_chart(hist_df, selected_ticker, period),
                        use_container_width=True, config={"displayModeBar": True})

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(rsi_chart(hist_df),  use_container_width=True, config={"displayModeBar": False})
        with col2:
            st.plotly_chart(macd_chart(hist_df), use_container_width=True, config={"displayModeBar": False})

        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(bb_width_chart(hist_df), use_container_width=True, config={"displayModeBar": False})
        with col4:
            st.plotly_chart(atr_chart(hist_df),      use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════
# TAB 3: INVESTMENT DECISION
# ══════════════════════════════════════════════════════════════
with tab3:
    signal_full, action, combined_score = investment_decision(f_score, t_score)

    # Key Takeaway
    takeaway_box(
        f"Combined signal for {name}: <b>{signal_full}</b> (score {combined_score}/100). "
        f"Fundamental Score: {f_score}/100 (60% weight) · Technical Score: {t_score}/100 (40% weight). "
        f"Composite Fair Value: {composite_str} vs current {price_str} ({upside_str})."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(f_score, "Fundamental Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_chart(t_score, "Technical Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_chart(combined_score, "Combined Score"),
                        use_container_width=True, config={"displayModeBar": False})

    # Score explanation
    with st.expander("📐 How are these scores calculated?", expanded=False):
        st.markdown("""
**Fundamental Score (0–100)** starts at 50 (neutral), then adjusts for:
- **Valuation gap** (±25 pts): how far current price is from Composite Fair Value
- **ROE** (+5 if >20%, +2 if >12%, -5 if negative)
- **Net Margin** (+5 if >15%, +2 if >8%, -5 if negative)
- **Debt/Equity** (+5 if <0.5x, -5 if >2x)
- **Current Ratio** (+3 if >1.5, -3 if <1)
- **Revenue Growth** (+5 if >10%, +2 if positive, -3 if <-5%)
- **Short Interest** (-5 if >20%, -2 if >10%)

**Technical Score (0–100)** starts at 50 (neutral), then adjusts for:
- **Price vs SMA200** (+12 if above, -8 if below) — primary trend indicator
- **Price vs SMA50** (+8 if above, -5 if below) — medium-term trend
- **EMA9 vs SMA50** (+5 if above, -3 if below) — short-term momentum
- **RSI** (+10 if <30/oversold, -10 if >70/overbought, +5 if 40–65/healthy)
- **MACD** (+8 if bullish crossover, -5 if bearish)
- **Proximity to S/R** (±5 if within 3% of support/resistance)

**Combined Score** = Fundamental × 60% + Technical × 40%
""")

    action_colors      = {"BUY": "#15803d", "ACCUMULATE": "#166534", "HOLD": "#854d0e", "REDUCE": "#9a3412", "SELL": "#7f1d1d"}
    action_text_colors = {"BUY": "#dcfce7", "ACCUMULATE": "#bbf7d0", "HOLD": "#fef9c3", "REDUCE": "#ffedd5", "SELL": "#fee2e2"}
    bg = action_colors.get(action, "#334155")
    tc = action_text_colors.get(action, "#f1f5f9")

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:24px;text-align:center;margin:16px 0">
      <div style="font-size:2.5rem;font-weight:800;color:{tc}">{signal_full}</div>
      <div style="color:{tc};opacity:0.8;font-size:1rem;margin-top:4px">
        Combined Score: {combined_score}/100 &nbsp;|&nbsp; Fundamental 60% + Technical 40%
      </div>
    </div>""", unsafe_allow_html=True)

    # Tranche Plan
    if current_price and hist_df is not None and not hist_df.empty:
        resistance_lvls, support_lvls = find_support_resistance(hist_df)
        plan, stop_loss, s1, s2, r1, r2 = tranche_plan(action, current_price, support_lvls, resistance_lvls)

        st.markdown("### 📋 Tranche Deployment Plan")
        if plan:
            plan_df = pd.DataFrame(plan)
            st.dataframe(plan_df, use_container_width=True)
        else:
            st.info("Tranche plan not available for current signal.")

        st.markdown("### ⚠️ Risk Metrics")
        r1c, r2c, r3c, r4c = st.columns(4)

        downside_to_s2  = ((s2 - current_price) / current_price * 100) if current_price else None
        analyst_target  = fund_data.get("analyst_target")
        upside_analyst  = ((analyst_target - current_price) / current_price * 100) if analyst_target and current_price else None
        upside_dcf_val  = ((dcf_val - current_price) / current_price * 100) if dcf_val and current_price else None
        best_upside     = upside_analyst if upside_analyst is not None else upside_dcf_val
        rr_ratio        = (
            round(best_upside / abs(downside_to_s2), 2)
            if best_upside is not None and downside_to_s2 and downside_to_s2 != 0
            else None
        )

        with r1c:
            ds = f"{downside_to_s2:.1f}%" if downside_to_s2 is not None else "N/A"
            metric_card("Downside to S2", ds, f"Support at ${s2:.2f}")
        with r2c:
            us = f"{upside_analyst:+.1f}%" if upside_analyst is not None else "N/A"
            metric_card("Upside to Analyst Target", us, fmt_currency(analyst_target))
        with r3c:
            us2 = f"{upside_dcf_val:+.1f}%" if upside_dcf_val is not None else "N/A"
            metric_card("Upside to FCF Value (DCF)", us2, fmt_currency(dcf_val))
        with r4c:
            rr       = f"{rr_ratio:.1f}x" if rr_ratio is not None else "N/A"
            rr_color = "#22c55e" if (rr_ratio and rr_ratio >= 2) else ("#ef4444" if (rr_ratio and rr_ratio < 1) else "#eab308")
            st.markdown(f"""<div class="metric-card" style="border-left:3px solid {rr_color}">
              <div class="metric-label">Risk / Reward Ratio</div>
              <div class="metric-value" style="color:{rr_color}">{rr}</div>
              <div class="metric-sub">Target: ≥ 2x for good trades</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #ef4444;border-radius:8px;padding:12px 16px;margin-top:8px">
          <span style="color:#94a3b8;font-size:0.85rem">🛑 Suggested Stop Loss: </span>
          <span style="color:#ef4444;font-weight:700;font-size:1rem">${stop_loss:.2f}</span>
          <span style="color:#64748b;font-size:0.8rem;margin-left:8px">
            ({((stop_loss-current_price)/current_price*100):.1f}% from current)
          </span>
        </div>""", unsafe_allow_html=True)

    # Risk Flags
    st.markdown("### 🚩 Risk Flags")
    flags = []
    de     = fund_data.get("debt_equity");  cr = fund_data.get("current_ratio")
    si     = fund_data.get("short_pct");    margin = fund_data.get("profit_margin")
    roe_v  = fund_data.get("roe")

    if de and de > 2:        flags.append(("risk", f"⚠️ High Debt/Equity: {de:.2f}x (>2 elevated)"))
    elif de and de < 0.5:    flags.append(("ok",   f"✅ Low Debt/Equity: {de:.2f}x (strong balance sheet)"))
    if cr and cr < 1:        flags.append(("risk", f"⚠️ Low Current Ratio: {cr:.2f} (liquidity concern)"))
    elif cr and cr > 2:      flags.append(("ok",   f"✅ Strong Current Ratio: {cr:.2f}"))
    if si and si > 0.15:     flags.append(("risk", f"⚠️ High Short Interest: {si*100:.1f}%"))
    if margin and margin < 0: flags.append(("risk", f"⚠️ Negative Net Margin: {margin*100:.1f}%"))
    elif margin and margin > 0.15: flags.append(("ok", f"✅ Strong Margin: {margin*100:.1f}%"))
    if roe_v and roe_v < 0:  flags.append(("risk", f"⚠️ Negative ROE: {roe_v*100:.1f}%"))
    elif roe_v and roe_v > 0.15: flags.append(("ok", f"✅ Strong ROE: {roe_v*100:.1f}%"))

    if not flags:
        st.markdown('<div class="ok-flag">✅ No major risk flags detected</div>', unsafe_allow_html=True)
    for flag_type, msg in flags:
        st.markdown(f'<div class="{"risk-flag" if flag_type=="risk" else "ok-flag"}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: UNIVERSE BROWSER
# ══════════════════════════════════════════════════════════════
with tab4:
    takeaway_box(
        "Browse all stocks in an index with fair valuations computed using the <b>Aggressive</b> weight preset "
        "(DCF 50%, PE 25%, Market Value 25%). Load up to 20 stocks at a time to stay within Yahoo Finance rate limits. "
        "Click <b>🔍 Analyze</b> in the sidebar for full deep-dive on any stock."
    )

    st.markdown("### 🏦 Stock Universe Browser")

    uni_choice      = st.selectbox("Select Universe", ["NASDAQ 100", "S&P 100", "NIFTY 100"], key="universe_browse")
    tickers_to_show = UNIVERSE_MAP[uni_choice]

    col_left, col_right = st.columns([2, 1])
    with col_left:
        batch_size = st.slider("Stocks to load", 5, 30, 15, 5, key="batch_size",
                               help="Yahoo Finance rate-limits large batch requests. Keep ≤ 20 for reliability.")
    with col_right:
        start_idx = st.number_input("Start from #", 1, len(tickers_to_show), 1, key="start_idx")

    batch = tickers_to_show[start_idx - 1 : start_idx - 1 + batch_size]
    load_btn = st.button(f"📥 Load {len(batch)} stocks", type="primary")

    if load_btn or st.session_state.get("uni_df") is not None:
        if load_btn:
            import time
            rows = []
            progress = st.progress(0, text="Loading stock data...")

            for i, (sym, cname) in enumerate(batch):
                progress.progress((i + 1) / len(batch), text=f"Fetching {sym}...")
                try:
                    import yfinance as yf
                    if i > 0 and i % 8 == 0:
                        time.sleep(2)   # brief rate-limit pause every 8 stocks

                    t    = yf.Ticker(sym)
                    info = t.info or {}

                    if len(info) < 10:
                        time.sleep(3)
                        info = yf.Ticker(sym).info or {}

                    price   = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                    prev    = info.get("previousClose") or price
                    chg_pct = ((price - prev) / prev * 100) if prev else 0

                    sec  = info.get("sector", "default")
                    sm   = fetch_sector_multiples(sec)
                    eps_ = info.get("trailingEps")
                    feps = info.get("forwardEps")
                    bvps_= info.get("bookValue")
                    fcf_ = info.get("freeCashflow")
                    ebi_ = info.get("ebitda")
                    shr_ = info.get("sharesOutstanding")
                    tdt  = info.get("totalDebt") or 0
                    csh  = info.get("totalCash") or 0
                    nd_  = tdt - csh

                    g_   = graham_number(eps_, bvps_)
                    d_   = dcf_valuation(fcf_, dcf_growth, shr_) if (fcf_ and fcf_ > 0 and shr_) else None
                    pv_, _ = pe_based_valuation(eps_, feps, sm["pe"])
                    ev_  = ev_ebitda_valuation(ebi_, nd_, shr_, sm["ev_ebitda"])
                    pb_  = pb_valuation(bvps_, sm["pb"])

                    cfv  = composite_fair_value(
                        {"graham": g_, "dcf": d_, "pe": pv_, "ev": ev_, "pb": pb_},
                        UNIVERSE_WEIGHTS
                    )

                    upside_ = ((cfv - price) / price * 100) if (cfv and price > 0) else None
                    sig_    = ("🟢 Buy" if upside_ and upside_ > 15 else
                               ("🔴 Sell" if upside_ and upside_ < -15 else "🟡 Hold")) if upside_ is not None else "N/A"

                    rec_  = (info.get("recommendationKey") or "").upper().replace("_", " ") or "N/A"
                    tgt_  = info.get("targetMeanPrice")
                    roe_  = info.get("returnOnEquity")
                    mgn_  = info.get("profitMargins")
                    rg_   = info.get("revenueGrowth")
                    roce_ = fund_data.get("roce") if sym == selected_ticker else None

                    # Industry entry/exit logic:
                    # Entry = buy on pullback to ~S1 if overvalued; else current price
                    # Exit = analyst target or estimated R1 (current × 1.08)
                    if upside_ is not None and upside_ > 5:
                        entry_ = f"${price:.2f} ✅"   # currently attractive
                    elif upside_ is not None and upside_ < -5:
                        entry_ = f"${price*0.92:.2f} (wait -8%)"  # wait for pullback
                    else:
                        entry_ = f"${price:.2f}"

                    exit_  = f"${tgt_:.2f}" if tgt_ else f"${price*1.10:.2f} (~+10%)"

                    rows.append({
                        "Ticker":        sym,
                        "Company":       cname[:22],
                        "Price ($)":     round(price, 2) if price else None,
                        "Chg %":         round(chg_pct, 2),
                        "Fair Value ($)": round(cfv, 2) if cfv else None,
                        "Upside %":      round(upside_, 1) if upside_ is not None else None,
                        "Signal":        sig_,
                        "Analyst Rec":   rec_,
                        "Entry":         entry_,
                        "Exit":          exit_,
                        "ROE %":         round(roe_ * 100, 1) if roe_ else None,
                        "Net Margin %":  round(mgn_ * 100, 1) if mgn_ else None,
                        "Rev Growth %":  round(rg_ * 100, 1) if rg_ else None,
                        "Sector":        sec if sec != "default" else "—",
                    })
                except Exception as ex:
                    rows.append({"Ticker": sym, "Company": cname, "Signal": "Error"})

            progress.empty()
            uni_df = pd.DataFrame(rows)
            st.session_state["uni_df"] = uni_df.to_dict("records")
        else:
            uni_df = pd.DataFrame(st.session_state["uni_df"])

        if not uni_df.empty and "Price ($)" in uni_df.columns:
            # Color-code upside column
            def _style_upside(val):
                if not isinstance(val, (int, float)): return ""
                return "color: #22c55e" if val > 15 else ("color: #ef4444" if val < -15 else "color: #eab308")

            styled = uni_df.style.applymap(_style_upside, subset=["Upside %"])
            st.dataframe(styled, use_container_width=True, height=520)

            # Summary stats
            buys  = sum(1 for r in uni_df.get("Signal", []) if "Buy" in str(r))
            sells = sum(1 for r in uni_df.get("Signal", []) if "Sell" in str(r))
            st.caption(f"📊 Of {len(uni_df)} stocks loaded: 🟢 {buys} Buy signals · 🔴 {sells} Sell signals · "
                       f"🟡 {len(uni_df)-buys-sells} Hold. Fair values use Aggressive weights (DCF 50% / PE 25% / Market Value 25%).")
        else:
            st.warning("Could not load data. Yahoo Finance may be rate-limiting. Wait 30–60s and retry.")

    st.markdown("---")
    st.markdown(f"**Total stocks in {uni_choice}:** {len(tickers_to_show)}")
    jump_options = [f"{sym} — {nm}" for sym, nm in tickers_to_show]
    jump_sel = st.selectbox("🔍 Quick-load for full analysis", ["— select —"] + jump_options)
    if jump_sel != "— select —":
        st.info(f"Select **{jump_sel.split(' — ')[0]}** in the sidebar (or type under Custom Ticker) to analyze it.")


# ══════════════════════════════════════════════════════════════
# TAB 5: GLOSSARY
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📖 Glossary of Terms")
    st.markdown("All financial terms and indicators used in this app, explained plainly.")

    def gloss(term, definition):
        st.markdown(f'<div class="gloss-term">{term}</div><div class="gloss-def">{definition}</div>',
                    unsafe_allow_html=True)

    st.markdown("### 💰 Equity Fair Valuation Methods")
    gloss("Safe Value (Graham Number)",
          "√(22.5 × EPS × Book Value Per Share). Benjamin Graham's intrinsic value formula. "
          "Conservative — built for value stocks, ignores future growth potential. "
          "High-growth companies almost always show as 'overvalued' by this method.")
    gloss("Future Cash Flow Value (DCF — Discounted Cash Flow)",
          "Projects Free Cash Flow 5 years forward at your chosen growth rate, discounts back at 10%, "
          "adds a terminal value (3% perpetual growth). Divides by shares for per-share intrinsic value. "
          "Most growth-sensitive method — small changes in growth rate have large impact.")
    gloss("Price to Earnings (PE-Based)",
          "EPS × Sector Benchmark P/E Multiple. Uses sector-median P/E (NOT the stock's own P/E — "
          "that would be circular). Falls back to Forward EPS when trailing EPS is negative. "
          "The benchmark P/E is a Damodaran/Bloomberg industry estimate.")
    gloss("Market Value (EV/EBITDA)",
          "Enterprise Value ÷ EBITDA at sector median multiple. EV = Market Cap + Debt − Cash. "
          "Capital-structure-neutral. Good for capital-intensive or debt-heavy industries.")
    gloss("Book Value (P/B)",
          "Book Value Per Share × Sector Median P/B multiple. Most relevant for asset-heavy sectors "
          "(banks, real estate). Less meaningful for intangible-heavy tech companies.")
    gloss("Dividends (DDM — Dividend Discount Model)",
          "D ÷ (Discount Rate − Dividend Growth). Gordon Growth Model. "
          "Applies only to dividend-paying stocks. Uses 4% fixed annual dividend growth (industry standard "
          "for mature payers). N/A for non-dividend stocks.")
    gloss("Composite Fair Value",
          "Weighted average of all applicable valuation methods. Methods returning N/A are excluded and "
          "remaining weights are auto-normalised. Adjust the Risk Appetite profile in the sidebar to "
          "change which methods carry more weight.")

    st.markdown("### 🎯 Risk Appetite Profiles")
    gloss("Aggressive",
          "Safe Value (Graham), Book Value, and Dividends have 0% weight — these anchor conservative "
          "investors but penalise fast-growing companies. DCF gets 50% (future growth matters most), "
          "PE and Market Value split the remaining 50%. Best for high-growth, pre-dividend companies.")
    gloss("Moderate",
          "Balanced: DCF (40%) leads, PE (25%) and Market Value (20%) support it, "
          "with a small Graham buffer (10%) and P/B (5%). A sensible default for most stocks.")
    gloss("Conservative",
          "All 6 methods are active. Graham (10%), Book Value (5%), Dividends (5%) provide a "
          "safety floor. DCF, PE, and Market Value share the remaining 80% equally. "
          "Best for mature, dividend-paying, low-debt companies.")

    st.markdown("### 📐 Key Financial Ratios")
    gloss("PEG Ratio",    "P/E ÷ Earnings Growth Rate. <1 = growth justifies price. >1.5 = expensive relative to growth.")
    gloss("Debt/Equity",  "Total Debt ÷ Equity. <0.5 = conservative. >2 = high leverage.")
    gloss("Current Ratio","Current Assets ÷ Current Liabilities. >1.5 = good liquidity. <1 = concern.")
    gloss("Interest Coverage", "EBIT ÷ Interest Expense. How many times interest is covered. <1.5x = stress risk.")
    gloss("ROE",          "Net Income ÷ Equity. >15% strong. >20% excellent. Negative = capital destruction.")
    gloss("ROA",          "Net Income ÷ Total Assets. Profitability relative to asset base.")
    gloss("ROCE",         "EBIT ÷ (Total Assets − Current Liabilities). Returns on long-term capital employed.")
    gloss("Net Margin",   "Net Income ÷ Revenue. >15% excellent. >8% healthy. Negative = unprofitable.")
    gloss("EBITDA Margin","EBITDA ÷ Revenue. Operating profitability before interest, tax, depreciation.")
    gloss("FCF Yield",    "Free Cash Flow ÷ Market Cap. >5% generally attractive. Higher = more cash-generative.")
    gloss("Beta",         "Sensitivity to market moves. >1.5 = high volatility. <0.7 = defensive.")
    gloss("Short Interest","% of float sold short. >15% = significant bearish bets. Can cause short squeezes.")
    gloss("EPS (Trailing)","Actual earnings per share over last 12 months.")
    gloss("EPS (Forward)", "Analyst-estimated earnings per share for next 12 months.")

    st.markdown("### 📈 Technical Indicators")
    gloss("SMA (Simple Moving Average)",     "Average closing price over N days. SMA50 and SMA200 are institutional benchmarks.")
    gloss("EMA (Exponential Moving Average)","Like SMA but weights recent prices more heavily. Reacts faster to moves.")
    gloss("RSI (Relative Strength Index)",   "0–100 momentum gauge. >70 = overbought. <30 = oversold. 40–65 = healthy uptrend.")
    gloss("MACD",                            "12-day EMA minus 26-day EMA. Bullish when crossing above signal line (9-day EMA).")
    gloss("Bollinger Bands",                 "SMA20 ± 2 standard deviations. BB Width squeeze = volatility about to expand.")
    gloss("ATR (Average True Range)",        "Daily price range averaged over 14 periods. Measures raw volatility in dollar terms.")
    gloss("Support Level",                   "Price where buyers have historically stepped in. S1 = nearest below current price.")
    gloss("Resistance Level",               "Price where sellers have historically appeared. R1 = nearest above current price.")

    st.markdown("### 🎯 Investment Decision Terms")
    gloss("Fundamental Score (0–100)","Valuation gap + ROE + margins + leverage + growth + short interest. >70 = bullish.")
    gloss("Technical Score (0–100)","Price trends + RSI + MACD + support/resistance proximity. >60 = bullish.")
    gloss("Combined Score",         "Fundamental × 60% + Technical × 40%. >70=BUY, 57-70=ACCUMULATE, 43-57=HOLD, 30-43=REDUCE, <30=SELL.")
    gloss("Tranche / Phased Buying","Split entry into 3 tranches (30/30/40%) at current, S1, and S2 levels to average cost down.")
    gloss("Stop Loss",              "Exit price to cap losses. Set below S2. Never risk more than 5–7% of capital per position.")
    gloss("Risk/Reward Ratio",      "Upside potential ÷ Downside risk. Professional traders require ≥ 2:1 before entry.")
    gloss("Entry Point",            "Optimal price to initiate or add to a position. 1st tranche = now if BUY signal.")
    gloss("Exit Point",             "Target sell price — typically analyst consensus target or key resistance level.")

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:0.75rem">Educational summaries only. Not a substitute for professional financial advice.</div>',
                unsafe_allow_html=True)
