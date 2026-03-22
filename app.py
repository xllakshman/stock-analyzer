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
# CUSTOM CSS — dark theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f172a; color: #f1f5f9; }
  [data-testid="stSidebar"] { background-color: #1e293b; }
  [data-testid="stSidebar"] .stMarkdown { color: #f1f5f9; }

  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .metric-label { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; }
  .metric-value { color: #f1f5f9; font-size: 1.35rem; font-weight: 600; margin-top: 2px; }
  .metric-sub   { color: #64748b; font-size: 0.78rem; margin-top: 2px; }

  .badge-buy    { background:#15803d; color:#dcfce7; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-accum  { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-hold   { background:#854d0e; color:#fef9c3; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-reduce { background:#9a3412; color:#ffedd5; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-sell   { background:#7f1d1d; color:#fee2e2; padding:3px 10px; border-radius:4px; font-weight:600; }

  .header-bar {
    background: #1e293b;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 16px;
    border-left: 4px solid #3b82f6;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Tab: active = blue text only, no background box */
  .stTabs [data-baseweb="tab-list"] {
    background: #1e293b; border-radius: 8px; padding: 4px; gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #94a3b8; border-radius: 6px; font-weight: 500;
    background: transparent !important;
  }
  .stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #3b82f6 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #3b82f6 !important;
  }

  .stDataFrame { background: #1e293b !important; }
  .stTextInput input, .stSelectbox select { background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
  .stNumberInput input { background: #1e293b !important; color: #f1f5f9 !important; }
  hr { border-color: #334155; }
  .progress-container { background:#334155; border-radius:4px; height:8px; overflow:hidden; }
  .progress-fill      { height:8px; border-radius:4px; }
  .stProgress > div > div { background-color: #3b82f6 !important; }
  .risk-flag { background:#7f1d1d; color:#fca5a5; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .ok-flag   { background:#14532d; color:#86efac; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .source-live { color:#22c55e; font-size:0.72rem; }
  .source-est  { color:#f59e0b; font-size:0.72rem; }

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
    source_html = ""
    if source:
        cls = "source-live" if "✅" in source else "source-est"
        source_html = f'<span class="{cls}">{source}</span>'
    if value is None:
        diff_html = '<span style="color:#64748b">N/A</span>'
        val_str   = "N/A"
    else:
        val_str = f"${value:,.2f}"
        if current_price:
            diff_pct = (value - current_price) / current_price * 100
            color  = "#22c55e" if diff_pct > 5 else ("#ef4444" if diff_pct < -5 else "#eab308")
            arrow  = "▲" if diff_pct > 0 else "▼"
            diff_html = f'<span style="color:{color}">{arrow} {abs(diff_pct):.1f}% {"upside" if diff_pct>0 else "downside"}</span>'
        else:
            diff_html = ""
    st.markdown(f"""
    <div class="metric-card" style="border-left:3px solid #3b82f6;">
      <div class="metric-label">{label} &nbsp;{source_html}</div>
      <div class="metric-value">{val_str}</div>
      <div class="metric-sub">{diff_html}</div>
    </div>""", unsafe_allow_html=True)


def analyst_rationale_bullets(fund_data, current_price):
    """
    Generate data-driven bullet points explaining the analyst recommendation.
    Based purely on quantitative signals available from Yahoo Finance.
    """
    bullets = []

    # Price target vs current
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
            bullets.append(f"🔴 Consensus target ${target:.2f}{cov} implies {sign}{upside:.1f}% downside — limited price upside priced in")
        if lo and hi:
            bullets.append(f"📊 Analyst target range: ${lo:.2f} – ${hi:.2f} (low/high spread of {(hi-lo)/target*100:.0f}% around mean)")

    # Revenue growth
    rev_g = fund_data.get("revenue_growth")
    if rev_g is not None:
        if rev_g > 0.20:
            bullets.append(f"🟢 Revenue growing {rev_g*100:.1f}% YoY — high-growth profile supports premium valuation")
        elif rev_g > 0.05:
            bullets.append(f"🟡 Revenue growing {rev_g*100:.1f}% YoY — moderate organic expansion")
        elif rev_g >= 0:
            bullets.append(f"🟡 Revenue growth is slow at {rev_g*100:.1f}% YoY — watch for acceleration")
        else:
            bullets.append(f"🔴 Revenue declining {abs(rev_g)*100:.1f}% YoY — top-line contraction is a red flag")

    # Earnings growth
    earn_g = fund_data.get("earnings_growth")
    if earn_g is not None:
        if earn_g > 0.25:
            bullets.append(f"🟢 Earnings growing {earn_g*100:.1f}% — strong EPS expansion can drive re-rating")
        elif earn_g < -0.10:
            bullets.append(f"🔴 Earnings contracting {abs(earn_g)*100:.1f}% — margin compression concern")

    # Profitability
    margin = fund_data.get("profit_margin")
    if margin is not None:
        if margin > 0.25:
            bullets.append(f"🟢 Net margin {margin*100:.1f}% — exceptional profitability & pricing power")
        elif margin > 0.12:
            bullets.append(f"🟡 Net margin {margin*100:.1f}% — healthy profitability")
        elif margin < 0:
            bullets.append(f"🔴 Net margin negative at {margin*100:.1f}% — company is not yet profitable")

    # ROE
    roe = fund_data.get("roe")
    if roe and roe > 0.15:
        bullets.append(f"🟢 ROE of {roe*100:.1f}% indicates efficient use of shareholder capital")

    # Balance sheet
    de = fund_data.get("debt_equity")
    if de is not None:
        if de < 0.3:
            bullets.append(f"🟢 Low leverage (D/E {de:.2f}x) — clean balance sheet with financial flexibility")
        elif de > 2.0:
            bullets.append(f"🔴 High debt load (D/E {de:.2f}x) — elevated financial risk in rising-rate environment")

    # Short interest
    si = fund_data.get("short_pct")
    if si and si > 0.15:
        bullets.append(f"🔴 Elevated short interest {si*100:.1f}% — significant institutional skepticism in the stock")
    elif si and si < 0.03:
        bullets.append(f"🟢 Low short interest {si*100:.1f}% — minimal bearish positioning")

    # Beta / volatility
    beta = fund_data.get("beta")
    if beta:
        if beta > 1.5:
            bullets.append(f"⚠️ High beta ({beta:.2f}) — stock moves more than market; higher risk/reward")
        elif beta < 0.7:
            bullets.append(f"🟢 Defensive beta ({beta:.2f}) — lower volatility relative to broad market")

    if not bullets:
        bullets.append("ℹ️ Insufficient quantitative data to generate detailed rationale for this ticker.")

    return bullets


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Analyzer")
    st.markdown("---")

    universe = st.selectbox(
        "📊 Stock Universe",
        ["Custom Ticker", "NASDAQ 100", "S&P 100", "NIFTY 100"],
        key="universe"
    )

    if universe == "Custom Ticker":
        ticker_input = st.text_input("Enter Ticker", value="AAPL", key="custom_ticker").upper().strip()
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
    dcf_growth = st.slider("DCF Growth Rate (%)", 0, 30, 10, 1) / 100

    # ── Composite Fair Value Weights ──
    st.markdown("---")
    st.markdown("#### ⚖️ Composite Weights")
    st.caption("Set 0 to exclude a method. Weights auto-normalise.")
    w_graham = st.slider("Graham Number",  0, 100, 15, 5, key="w_graham")
    w_dcf    = st.slider("DCF",            0, 100, 30, 5, key="w_dcf")
    w_pe     = st.slider("PE-Based",       0, 100, 25, 5, key="w_pe")
    w_ev     = st.slider("EV/EBITDA",      0, 100, 15, 5, key="w_ev")
    w_pb     = st.slider("Price/Book",     0, 100, 10, 5, key="w_pb")
    w_ddm    = st.slider("DDM",            0, 100,  5, 5, key="w_ddm")
    total_w  = w_graham + w_dcf + w_pe + w_ev + w_pb + w_ddm
    if total_w > 0:
        st.caption(f"Total: {total_w}  (effective %: G={w_graham*100//total_w}% D={w_dcf*100//total_w}% PE={w_pe*100//total_w}% EV={w_ev*100//total_w}% PB={w_pb*100//total_w}% DDM={w_ddm*100//total_w}%)")
    else:
        st.warning("Set at least one weight > 0")

    weights = {"graham": w_graham, "dcf": w_dcf, "pe": w_pe,
               "ev": w_ev, "pb": w_pb, "ddm": w_ddm}

    st.markdown("---")
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
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

# ── Rate-limit / empty-data guard ──────────────────────────────
_price_ok = fund_data.get("current_price") is not None
_data_ok  = bool(fund_data) and _price_ok

if not _data_ok:
    st.error(
        "⚠️ **Yahoo Finance returned no data for this ticker.**\n\n"
        "This is almost always a **rate-limit** from Yahoo Finance's servers — not a bug in the app. "
        "Streamlit Community Cloud shares its outbound IP with many other apps, and Yahoo Finance "
        "briefly blocks requests from that IP when too many arrive at once."
    )
    st.info(
        "**What to do:**\n"
        "1. Wait **30–60 seconds**, then click **🔍 Analyze** in the sidebar to retry.\n"
        "2. If it still fails after 2–3 retries, wait 2–3 minutes and try again.\n"
        "3. Try a different ticker first to 'warm up' the session, then return to this one."
    )
    st.stop()

# Header
current_price = fund_data.get("current_price")
name          = fund_data.get("name", selected_name)
sector        = fund_data.get("sector", "default")

# Fetch live sector multiples (1 hr cache, labeled in UI)
sector_mults = fetch_sector_multiples(sector)

# Compute 1-day change from history
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

    # ── COMPUTE FAIR VALUES with live/fallback sector multiples ──
    graham = graham_number(eps, bvps)
    dcf    = dcf_valuation(fcf, dcf_growth, shares)

    # PE-Based: uses forward EPS as fallback when trailing EPS is negative/unavailable
    pe_val, pe_eps_label = pe_based_valuation(eps, forward_eps, sector_mults["pe"])

    ev_val = ev_ebitda_valuation(ebitda, net_debt, shares, sector_mults["ev_ebitda"])
    pb_val = pb_valuation(bvps, sector_mults["pb"])

    # DDM: 4% fixed dividend growth rate (industry standard for mature dividend payers).
    # NOTE: fiveYearAvgDividendYield is a yield %, NOT a growth rate — not used here.
    ddm_growth = 0.04
    ddm        = dividend_discount_model(div_rate, ddm_growth)

    peg_v, peg_sig = peg_signal(pe, earnings_growth)

    # Weighted composite
    composite = composite_fair_value(
        {"graham": graham, "dcf": dcf, "pe": pe_val,
         "ev": ev_val, "pb": pb_val, "ddm": ddm},
        weights
    )
    signal_text, upside_pct = fundamental_signal(current_price, composite)

    metrics_for_score = {
        "roe":          fund_data.get("roe"),
        "net_margin":   fund_data.get("profit_margin"),
        "debt_equity":  fund_data.get("debt_equity"),
        "current_ratio": fund_data.get("current_ratio"),
        "revenue_growth": fund_data.get("revenue_growth"),
        "short_pct":    (fund_data.get("short_pct") or 0) * 100,
    }
    f_score = fundamental_score(current_price, composite, metrics_for_score)

    # ── SIGNAL BANNER ──
    banner_color = "#14532d" if "Undervalued" in signal_text else ("#7f1d1d" if "Overvalued" in signal_text else "#713f12")
    border_color = "#22c55e" if "Undervalued" in signal_text else ("#ef4444" if "Overvalued" in signal_text else "#eab308")
    composite_str = f"${composite:,.2f}" if composite else "N/A"
    upside_str    = f"{upside_pct:+.1f}%" if upside_pct is not None else "N/A"

    # Weight summary
    eff = {}
    if total_w > 0:
        eff = {k: int(v * 100 / total_w) for k, v in weights.items() if v > 0}
    weight_note = " · ".join(f"{k.upper()}:{v}%" for k, v in eff.items()) if eff else "—"

    st.markdown(f"""
    <div style="background:{banner_color};border:1px solid {border_color};border-radius:8px;padding:16px 20px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <span style="font-size:1.2rem;font-weight:700;color:#f1f5f9">{signal_text}</span>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">
            Composite Fair Value: <b style="color:#f1f5f9">{composite_str}</b> &nbsp;|&nbsp;
            Current Price: <b style="color:#f1f5f9">{price_str}</b> &nbsp;|&nbsp;
            Upside/Downside: <b style="color:{border_color}">{upside_str}</b>
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
    st.markdown("### Fair Value Methods")

    c1, c2, c3 = st.columns(3)
    with c1:
        valuation_card("Graham Number", graham, current_price,
                       "⚠️ est. (formula-based)")
    with c2:
        valuation_card(f"DCF (5-yr, {dcf_growth*100:.0f}% growth)", dcf, current_price,
                       "⚠️ est. (user growth rate)")
    with c3:
        pe_lbl = f"PE-Based ({sector_mults['pe']}x"
        if pe_eps_label:
            pe_lbl += f" × {pe_eps_label}"
        pe_lbl += ")"
        valuation_card(pe_lbl, pe_val, current_price, sector_mults["pe_source"])

    c4, c5, c6 = st.columns(3)
    with c4:
        valuation_card(f"EV/EBITDA ({sector_mults['ev_ebitda']}x)", ev_val, current_price,
                       sector_mults["ev_ebitda_source"])
    with c5:
        valuation_card(f"Price/Book ({sector_mults['pb']}x)", pb_val, current_price,
                       sector_mults["pb_source"])
    with c6:
        if div_rate and div_rate > 0:
            valuation_card("DDM (4% growth ⚠️ est.)", ddm, current_price,
                           "⚠️ est. (fixed 4% growth)")
        else:
            st.markdown("""
            <div class="metric-card" style="border-left:3px solid #334155">
              <div class="metric-label">Dividend Discount Model</div>
              <div class="metric-value" style="color:#64748b">N/A</div>
              <div class="metric-sub">Not a dividend-paying stock</div>
            </div>""", unsafe_allow_html=True)

    # Source legend
    st.markdown("""
    <div style="color:#64748b;font-size:0.75rem;margin-bottom:12px">
      ✅ live = fetched from sector ETF (XLK/XLF/etc.) &nbsp;·&nbsp;
      ⚠️ est. = industry-consensus fallback (Damodaran/Bloomberg sector medians, hardcoded)
    </div>""", unsafe_allow_html=True)

    # PEG — shown separately
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
              ⚠️ PEG and Composite Fair Value can disagree: PEG rewards fast-growing companies;
              absolute methods (Graham, DCF) may still flag them as expensive on a pure value basis — both views are valid.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── KEY METRICS ──
    st.markdown("### Key Fundamental Metrics")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**📈 Growth**")
        metric_card("Revenue Growth (YoY)",  fmt_pct(fund_data.get("revenue_growth")))
        metric_card("Revenue 3Y CAGR",       fmt_pct(fund_data.get("revenue_3yr_cagr")))
        metric_card("Earnings Growth (YoY)", fmt_pct(fund_data.get("earnings_growth")))
        metric_card("Earnings 3Y CAGR",      fmt_pct(fund_data.get("earnings_3yr_cagr")))
        metric_card("Free Cash Flow",         fmt_currency(fcf),
                    f"Yield: {fmt_pct(fund_data.get('fcf_yield'))}")

    with col_b:
        st.markdown("**💰 Profitability & Quality**")
        metric_card("EBITDA Margin",        fmt_pct(fund_data.get("ebitda_margin")))
        metric_card("Net Profit Margin",    fmt_pct(fund_data.get("profit_margin")))
        metric_card("Return on Equity (ROE)", fmt_pct(fund_data.get("roe")))
        metric_card("Return on Assets (ROA)", fmt_pct(fund_data.get("roa")))
        metric_card("ROCE (Approx.)",         fmt_pct(fund_data.get("roce")))

    with col_c:
        st.markdown("**⚖️ Risk & Ownership**")
        metric_card("Debt / Equity",       fmt_num(fund_data.get("debt_equity")))
        metric_card("Current Ratio",       fmt_num(fund_data.get("current_ratio")))
        metric_card("Interest Coverage",   fmt_num(fund_data.get("interest_coverage")))
        metric_card("Insider Ownership",   fmt_pct(fund_data.get("insider_ownership")))
        metric_card("Institutional Own.",  fmt_pct(fund_data.get("institutional_ownership")))

    col_d, col_e = st.columns(2)

    with col_d:
        st.markdown("**🎯 Analyst Consensus & Rationale**")

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
        rec_color = rec_colors.get(rec.replace(" ", "").upper(), rec_colors.get(rec, "#334155"))

        # Recommendation mean bar (1=Strong Buy … 5=Strong Sell)
        mean_bar_html = ""
        if rec_mean is not None:
            pct = max(0, min(100, (rec_mean - 1) / 4 * 100))
            bar_color = "#22c55e" if pct < 30 else ("#ef4444" if pct > 70 else "#eab308")
            mean_bar_html = f"""
            <div style="margin-top:6px">
              <div style="display:flex;justify-content:space-between;color:#64748b;font-size:0.7rem">
                <span>Strong Buy (1)</span><span>Hold (3)</span><span>Strong Sell (5)</span>
              </div>
              <div style="background:#334155;border-radius:4px;height:6px;overflow:hidden;margin:2px 0">
                <div style="background:{bar_color};width:{pct:.0f}%;height:6px;border-radius:4px"></div>
              </div>
              <div style="color:#94a3b8;font-size:0.75rem">Mean score: {rec_mean:.1f} / 5.0
                {'· ' + str(int(ana_count)) + ' analysts' if ana_count else ''}</div>
            </div>"""

        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Analyst Recommendation</div>
          <div class="metric-value" style="color:{rec_color}">{rec}</div>
          <div class="metric-sub">Avg Target: {fmt_currency(fund_data.get('analyst_target'))} &nbsp;|&nbsp;
          Low: {fmt_currency(fund_data.get('analyst_low'))} &nbsp;|&nbsp;
          High: {fmt_currency(fund_data.get('analyst_high'))}</div>
          {mean_bar_html}
        </div>""", unsafe_allow_html=True)

        metric_card("Beta (Market Risk)", fmt_num(fund_data.get("beta")),
                    "Beta >1 = more volatile than market")

        # Analyst rationale bullets
        if rec != "N/A":
            with st.expander("💡 Why this recommendation? (Data-Driven Rationale)", expanded=False):
                st.markdown(
                    '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:8px">'
                    'Auto-generated from quantitative signals. Not actual analyst notes.</div>',
                    unsafe_allow_html=True
                )
                for bullet in analyst_rationale_bullets(fund_data, current_price):
                    st.markdown(f"- {bullet}")

    with col_e:
        st.markdown("**📊 52-Week Range**")
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
            metric_card("Dividend Rate",  f"${div_rate:.2f}/yr",
                        f"Yield: {fmt_pct(fund_data.get('dividend_yield'))}")
        metric_card("Market Cap", fmt_currency(fund_data.get("market_cap")))
        metric_card("Short Interest %", fmt_pct(fund_data.get("short_pct")),
                    "High >15% is bearish pressure")


# ══════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if hist_df is None or hist_df.empty:
        st.error(f"No historical data available for {selected_ticker}.")
    else:
        t_score, t_signals = technical_score(hist_df)

        sig_text   = "🟢 Bullish" if t_score >= 60 else ("🔴 Bearish" if t_score < 40 else "🟡 Neutral")
        sig_color  = "#14532d" if t_score >= 60 else ("#7f1d1d" if t_score < 40 else "#713f12")
        sig_border = "#22c55e" if t_score >= 60 else ("#ef4444" if t_score < 40 else "#eab308")

        st.markdown(f"""
        <div style="background:{sig_color};border:1px solid {sig_border};border-radius:8px;
        padding:12px 20px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{sig_text}</span>
          <span style="color:{sig_border};font-size:1.5rem;font-weight:700">Technical Score: {t_score}%</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 Signal Breakdown", expanded=False):
            cols = st.columns(2)
            items = list(t_signals.items())
            for i, (k, v) in enumerate(items):
                cols[i % 2].markdown(f"**{k}**: {v}")

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
    f_score2 = f_score
    t_score2, _ = technical_score(hist_df) if (hist_df is not None and not hist_df.empty) else (50, {})
    signal_full, action, combined_score = investment_decision(f_score2, t_score2)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge_chart(f_score2, "Fundamental Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(gauge_chart(t_score2, "Technical Score"),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(gauge_chart(combined_score, "Combined Score"),
                        use_container_width=True, config={"displayModeBar": False})

    action_colors      = {"BUY": "#15803d", "ACCUMULATE": "#166534", "HOLD": "#854d0e", "REDUCE": "#9a3412", "SELL": "#7f1d1d"}
    action_text_colors = {"BUY": "#dcfce7", "ACCUMULATE": "#bbf7d0", "HOLD": "#fef9c3", "REDUCE": "#ffedd5", "SELL": "#fee2e2"}
    bg = action_colors.get(action, "#334155")
    tc = action_text_colors.get(action, "#f1f5f9")

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:24px;text-align:center;margin:16px 0">
      <div style="font-size:2.5rem;font-weight:800;color:{tc}">{signal_full}</div>
      <div style="color:{tc};opacity:0.8;font-size:1rem;margin-top:4px">
        Combined Score: {combined_score}% &nbsp;|&nbsp; Fundamental 60% + Technical 40%
      </div>
    </div>""", unsafe_allow_html=True)

    if current_price and hist_df is not None and not hist_df.empty:
        from utils.calculations import find_support_resistance
        resistance, support = find_support_resistance(hist_df)
        plan, stop_loss, s1, s2, r1, r2 = tranche_plan(action, current_price, support, resistance)

        st.markdown("### 📋 Tranche Deployment Plan")
        plan_df = pd.DataFrame(plan)
        st.dataframe(plan_df, use_container_width=True, hide_index=True)

        st.markdown("### ⚠️ Risk Metrics")
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
            ds = f"{downside_to_s2:.1f}%" if downside_to_s2 is not None else "N/A"
            metric_card("Downside to S2", ds, f"Strong support at ${s2:.2f}")
        with r2c:
            us = f"{upside_analyst:+.1f}%" if upside_analyst is not None else "N/A"
            metric_card("Upside to Analyst Target", us, fmt_currency(analyst_target))
        with r3c:
            us2 = f"{upside_dcf:+.1f}%" if upside_dcf is not None else "N/A"
            metric_card("Upside to DCF Value", us2, fmt_currency(dcf))
        with r4c:
            rr       = f"{rr_ratio:.1f}x" if rr_ratio is not None else "N/A"
            rr_color = "#22c55e" if (rr_ratio and rr_ratio >= 2) else ("#ef4444" if (rr_ratio and rr_ratio < 1) else "#eab308")
            st.markdown(f"""<div class="metric-card" style="border-left:3px solid {rr_color}">
              <div class="metric-label">Risk / Reward Ratio</div>
              <div class="metric-value" style="color:{rr_color}">{rr}</div>
              <div class="metric-sub">Target: ≥ 2x</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1e293b;border:1px solid #ef4444;border-radius:8px;padding:12px 16px;margin-top:8px">
          <span style="color:#94a3b8;font-size:0.85rem">🛑 Suggested Stop Loss: </span>
          <span style="color:#ef4444;font-weight:700;font-size:1rem">${stop_loss:.2f}</span>
          <span style="color:#64748b;font-size:0.8rem;margin-left:8px">
            ({((stop_loss - current_price)/current_price*100):.1f}% from current)
          </span>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🚩 Risk Flags")
    flags = []
    de     = fund_data.get("debt_equity")
    cr     = fund_data.get("current_ratio")
    si     = fund_data.get("short_pct")
    margin = fund_data.get("profit_margin")
    roe_v  = fund_data.get("roe")

    if de and de > 2:        flags.append(("risk", f"⚠️ High Debt/Equity: {de:.2f}x (>2 is elevated)"))
    elif de and de < 0.5:    flags.append(("ok",   f"✅ Low Debt/Equity: {de:.2f}x (strong balance sheet)"))
    if cr and cr < 1:        flags.append(("risk", f"⚠️ Low Current Ratio: {cr:.2f} (liquidity concern)"))
    elif cr and cr > 2:      flags.append(("ok",   f"✅ Strong Current Ratio: {cr:.2f}"))
    if si and si > 0.15:     flags.append(("risk", f"⚠️ High Short Interest: {si*100:.1f}% (bearish sentiment)"))
    if margin and margin < 0: flags.append(("risk", f"⚠️ Negative Net Margin: {margin*100:.1f}%"))
    elif margin and margin > 0.15: flags.append(("ok", f"✅ Strong Margin: {margin*100:.1f}%"))
    if roe_v and roe_v < 0:  flags.append(("risk", f"⚠️ Negative ROE: {roe_v*100:.1f}%"))
    elif roe_v and roe_v > 0.15: flags.append(("ok", f"✅ Strong ROE: {roe_v*100:.1f}%"))

    if not flags:
        st.markdown('<div class="ok-flag">✅ No major risk flags detected</div>', unsafe_allow_html=True)
    for flag_type, msg in flags:
        st.markdown(f'<div class="{"risk-flag" if flag_type=="risk" else "ok-flag"}">{msg}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: UNIVERSE BROWSER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏦 Stock Universe Browser")
    st.markdown("Browse all stocks in an index with key metrics.")

    uni_choice     = st.selectbox("Select Universe to Browse",
                                   ["NASDAQ 100", "S&P 100", "NIFTY 100"], key="universe_browse")
    tickers_to_show = UNIVERSE_MAP[uni_choice]
    show_all        = st.checkbox("Load all tickers (may take 2-3 minutes)", value=False)

    if show_all:
        with st.spinner(f"Fetching data for {len(tickers_to_show)} stocks..."):
            from utils.data import fetch_universe_snapshot
            df_universe = fetch_universe_snapshot(tickers_to_show)
    else:
        df_universe = pd.DataFrame([{"Ticker": sym, "Company": name} for sym, name in tickers_to_show])
        st.info("Check 'Load all tickers' above to fetch live prices and metrics for all stocks.")

    if "Change %" in df_universe.columns:
        df_styled = df_universe.style.format({
            "Price": "${:,.2f}", "Change %": "{:+.2f}%",
            "Market Cap ($B)": "${:.1f}B", "P/E": "{:.1f}",
        }).background_gradient(subset=["Change %"], cmap="RdYlGn", vmin=-5, vmax=5)
        st.dataframe(df_styled, use_container_width=True, height=500)
    else:
        st.dataframe(df_universe, use_container_width=True, height=500)

    st.markdown(f"**Total stocks in {uni_choice}:** {len(tickers_to_show)}")
    st.markdown("---")
    jump_options = [f"{sym} — {nm}" for sym, nm in tickers_to_show]
    jump_sel     = st.selectbox("🔍 Quick-load a stock for analysis", ["— select —"] + jump_options)
    if jump_sel != "— select —":
        jump_ticker = jump_sel.split(" — ")[0]
        st.info(f"Select **{jump_ticker}** in the sidebar (or type it under Custom Ticker) to analyze it.")


# ══════════════════════════════════════════════════════════════
# TAB 5: GLOSSARY
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📖 Glossary of Terms")
    st.markdown("All financial terms and indicators used in this app, explained plainly.")

    def gloss(term, definition):
        st.markdown(
            f'<div class="gloss-term">{term}</div>'
            f'<div class="gloss-def">{definition}</div>',
            unsafe_allow_html=True
        )

    # ── VALUATION ──
    st.markdown("### 💰 Valuation Methods")
    gloss("Graham Number",
          "Intrinsic value formula by Benjamin Graham: √(22.5 × EPS × Book Value Per Share). "
          "Combines earnings and assets into a single conservative price target. "
          "Best suited for value stocks; systematically shows growth stocks as overvalued because it ignores future growth.")
    gloss("DCF — Discounted Cash Flow",
          "Projects a company's Free Cash Flow (FCF) over 5 years at your chosen growth rate, then discounts future cash flows "
          "back to today's value using a 10% discount rate. Adds a terminal value for cash flows beyond year 5 (at 3% perpetual growth). "
          "Divides total by shares outstanding to get per-share intrinsic value. Most sensitive to growth rate assumption.")
    gloss("PE-Based Valuation",
          "EPS × Sector Benchmark P/E Multiple. Uses the sector's median P/E (fetched live from sector ETF or industry estimate) "
          "instead of the stock's own P/E — using the stock's own P/E would always produce the current price (circular math). "
          "Falls back to Forward EPS when trailing EPS is negative (loss-making companies).")
    gloss("EV/EBITDA Valuation",
          "Enterprise Value ÷ EBITDA is a capital-structure-neutral valuation ratio. Here: Intrinsic EV = EBITDA × Sector Median Multiple, "
          "then subtract Net Debt and divide by shares to get equity value per share. Good for capital-intensive industries.")
    gloss("Price-to-Book (P/B) Valuation",
          "Book Value Per Share × Sector Median P/B Multiple. Book value is assets minus liabilities on the balance sheet. "
          "Most useful for asset-heavy sectors (Financials, Real Estate). Less meaningful for intangible-heavy tech companies.")
    gloss("DDM — Dividend Discount Model (Gordon Growth)",
          "Intrinsic Value = Dividend Per Share ÷ (Discount Rate − Dividend Growth Rate). "
          "Uses a fixed 4% annual dividend growth rate assumption. Applies only to dividend-paying stocks. "
          "N/A for non-dividend payers or when dividend growth exceeds the discount rate.")
    gloss("Composite Fair Value",
          "Weighted average of all applicable valuation methods above. Methods returning N/A are excluded and remaining weights are "
          "auto-normalised to 100%. You can adjust individual weights in the sidebar (0 = exclude that method entirely).")
    gloss("Upside / Downside %",
          "(Composite Fair Value − Current Price) / Current Price × 100. "
          "Positive = stock trading below intrinsic value (upside potential). "
          "Negative = stock trading above intrinsic value (downside risk). "
          ">+15% = Undervalued  |  -15% to +15% = Fair Value  |  <-15% = Overvalued.")
    gloss("Intrinsic Value",
          "The estimated 'true' worth of a stock based on its fundamentals (earnings, cash flow, assets), "
          "independent of current market price. If market price < intrinsic value, the stock may be undervalued.")

    # ── RATIOS ──
    st.markdown("### 📐 Key Financial Ratios")
    gloss("P/E Ratio (Trailing)",
          "Price ÷ Trailing 12-month EPS. How much investors pay per dollar of past earnings. "
          "High P/E may mean overvalued or high growth expectations. Low P/E may mean cheap or declining earnings.")
    gloss("P/E Ratio (Forward)",
          "Price ÷ Next 12-month Estimated EPS (analyst consensus). More forward-looking than trailing P/E. "
          "Used in PE-Based valuation when trailing EPS is negative.")
    gloss("PEG Ratio",
          "P/E ÷ Earnings Growth Rate (%). Adjusts P/E for growth. "
          "PEG < 1: growth more than justifies valuation (undervalued on growth basis). "
          "PEG 1-1.5: fairly priced relative to growth. PEG > 1.5: expensive relative to growth. "
          "Note: PEG can disagree with absolute value methods (DCF, Graham) — "
          "PEG rewards high-growth companies; absolute methods may still flag them as expensive.")
    gloss("EV/EBITDA",
          "Enterprise Value ÷ EBITDA. EV = Market Cap + Debt − Cash. "
          "Capital-structure-neutral valuation multiple. Lower = cheaper. Useful for comparing companies across sectors.")
    gloss("Price/Book (P/B)",
          "Market Price ÷ Book Value Per Share. P/B < 1 means stock trades below its net asset value. "
          "High P/B is acceptable for high-ROE companies (e.g. tech). Very meaningful for banks and asset-heavy companies.")
    gloss("Debt/Equity (D/E)",
          "Total Debt ÷ Shareholders' Equity. Measures financial leverage. "
          "D/E < 0.5: conservative/low risk. D/E 0.5-2: moderate. D/E > 2: high leverage, elevated risk.")
    gloss("Current Ratio",
          "Current Assets ÷ Current Liabilities. Measures short-term liquidity. "
          "> 1.5: healthy liquidity. 1-1.5: adequate. < 1: may struggle to meet near-term obligations.")
    gloss("Interest Coverage",
          "EBIT ÷ Interest Expense. How many times the company can cover its interest payments from operating profit. "
          "> 3x: comfortable. < 1.5x: financial stress risk.")
    gloss("EPS — Earnings Per Share",
          "Net Income ÷ Shares Outstanding. The profit attributable to each share. "
          "Trailing EPS: actual past 12 months. Forward EPS: analyst estimate for next 12 months.")
    gloss("Book Value Per Share",
          "Total Shareholders' Equity ÷ Shares Outstanding. Net asset value per share on the balance sheet.")

    # ── PROFITABILITY ──
    st.markdown("### 📊 Profitability Metrics")
    gloss("ROE — Return on Equity",
          "Net Income ÷ Shareholders' Equity. Measures how efficiently management generates profit from equity capital. "
          "> 15%: strong. > 20%: excellent. Negative = destroying shareholder value.")
    gloss("ROA — Return on Assets",
          "Net Income ÷ Total Assets. Shows profitability relative to total assets employed. "
          "More meaningful for asset-heavy industries (manufacturing, finance).")
    gloss("ROCE — Return on Capital Employed",
          "EBIT ÷ (Total Assets − Current Liabilities). Measures return on long-term capital. "
          "A company with ROCE > its cost of capital is creating value.")
    gloss("Net Profit Margin",
          "Net Income ÷ Revenue. The percentage of revenue that becomes profit after all expenses. "
          "> 15%: excellent. > 8%: healthy. Negative: unprofitable.")
    gloss("EBITDA Margin",
          "EBITDA ÷ Revenue. Pre-interest, pre-tax, pre-depreciation operating margin. "
          "Strips out capital structure and accounting differences, useful for cross-company comparison.")
    gloss("Free Cash Flow (FCF)",
          "Operating Cash Flow − Capital Expenditures. The cash a company generates after funding its operations and investments. "
          "Companies with high FCF can pay dividends, buy back shares, or invest in growth.")
    gloss("FCF Yield",
          "FCF ÷ Market Cap. The 'earnings yield' equivalent using cash flow. "
          "Higher = more cash-generative relative to valuation. > 5% is generally attractive.")

    # ── GROWTH ──
    st.markdown("### 📈 Growth Metrics")
    gloss("Revenue Growth (YoY)",
          "Year-over-year change in total revenue. Measures top-line expansion. "
          "> 10%: strong growth. 0-10%: moderate. Negative: contraction (concerning).")
    gloss("Revenue 3Y CAGR",
          "Compound Annual Growth Rate of revenue over 3 years. Smooths out year-to-year volatility; "
          "better reflects the structural growth trend.")
    gloss("Earnings Growth (YoY)",
          "Year-over-year change in net income or EPS. Drives stock re-rating when sustained. "
          "Earnings growing faster than revenue signals margin expansion (positive).")
    gloss("Earnings 3Y CAGR",
          "3-year compound annual growth in earnings. A consistent 15%+ CAGR is what most growth investors target.")

    # ── TECHNICAL ──
    st.markdown("### 📈 Technical Indicators")
    gloss("SMA — Simple Moving Average",
          "Average closing price over N periods (SMA50 = last 50 days, SMA200 = last 200 days). "
          "Price above SMA200 = long-term uptrend. Price below = downtrend. "
          "SMA200 is the most-watched line by institutional traders.")
    gloss("EMA — Exponential Moving Average",
          "Like SMA but gives more weight to recent prices, so it reacts faster to price changes. "
          "EMA9 crossing above SMA50 is a short-term bullish signal.")
    gloss("RSI — Relative Strength Index",
          "Momentum oscillator scaled 0–100. Measures speed and magnitude of price changes. "
          "RSI > 70: overbought (potential reversal down). RSI < 30: oversold (potential reversal up). "
          "RSI 40–65: neutral-to-bullish momentum zone.")
    gloss("MACD — Moving Average Convergence Divergence",
          "Difference between 12-day EMA and 26-day EMA. Signal line is 9-day EMA of MACD. "
          "MACD crossing above signal line = bullish. Below signal = bearish. "
          "Histogram shows the gap between MACD and signal.")
    gloss("Bollinger Bands",
          "SMA20 ± (2 × standard deviation of price). Bands expand in volatile markets, contract in calm ones. "
          "BB Width = (Upper − Lower) / Middle Band. Narrow band (squeeze) often precedes a big price move.")
    gloss("ATR — Average True Range",
          "Average of the daily high-low range (adjusted for gaps) over 14 periods. "
          "Measures volatility in dollar terms. Higher ATR = bigger daily swings. "
          "Useful for setting stop-loss levels: typically 1.5–2× ATR below entry.")
    gloss("Support Level",
          "Price level where buying demand has historically stopped a decline. "
          "S1 = nearest support below current price; S2 = secondary (deeper) support.")
    gloss("Resistance Level",
          "Price level where selling pressure has historically capped a rally. "
          "R1 = nearest resistance above current price; R2 = next higher resistance.")
    gloss("Golden Cross / Death Cross",
          "Golden Cross: SMA50 crosses above SMA200 — long-term bullish signal. "
          "Death Cross: SMA50 crosses below SMA200 — long-term bearish signal.")

    # ── MARKET / RISK ──
    st.markdown("### ⚖️ Market & Risk Metrics")
    gloss("Market Capitalisation",
          "Share Price × Shares Outstanding. Total market value of the company. "
          "Large-cap > $10B. Mid-cap $2B–$10B. Small-cap < $2B.")
    gloss("Beta",
          "Sensitivity of the stock to market movements (relative to S&P 500). "
          "Beta 1.0 = moves with market. Beta 1.5 = 50% more volatile. Beta 0.5 = half as volatile. "
          "Negative beta = moves opposite to market (rare).")
    gloss("Short Interest %",
          "Shares sold short as % of float (shares available to trade). "
          "High short interest (> 10–15%) means many investors are betting the stock will fall. "
          "Can also lead to a 'short squeeze' if the stock rises sharply.")
    gloss("Insider Ownership",
          "% of shares held by company directors, executives, and major shareholders (>10%). "
          "High insider ownership aligns management interests with shareholders.")
    gloss("Institutional Ownership",
          "% of shares held by hedge funds, mutual funds, pension funds. "
          "High institutional ownership signals professional confidence in the stock.")
    gloss("Analyst Recommendation Mean",
          "Average of all analyst ratings on a 1–5 scale: 1=Strong Buy, 2=Buy, 3=Hold, 4=Underperform, 5=Sell. "
          "A mean below 2.0 indicates strong bullish consensus.")
    gloss("52-Week Range",
          "The lowest and highest price the stock has traded at in the past 52 weeks. "
          "The bar shows where the current price sits within that range. "
          "Near 52-week lows can indicate value; near highs can indicate momentum or overvaluation.")

    # ── INVESTMENT DECISION ──
    st.markdown("### 🎯 Investment Decision Terms")
    gloss("Fundamental Score (0–100)",
          "Composite score based on: valuation gap to fair value (±25 pts), ROE, net margin, debt/equity, "
          "current ratio, revenue growth, and short interest. > 70 = bullish fundamentals.")
    gloss("Technical Score (0–100)",
          "Composite score based on: price vs SMA50/SMA200, EMA9 vs SMA50, RSI level, MACD crossover, "
          "and proximity to key support/resistance. > 60 = bullish technicals.")
    gloss("Combined Score",
          "Fundamental Score × 60% + Technical Score × 40%. "
          "> 70 = BUY. 57–70 = ACCUMULATE. 43–57 = HOLD. 30–43 = REDUCE. < 30 = SELL.")
    gloss("Tranche / Phased Buying",
          "Splitting a position into multiple buy entries instead of investing all at once. "
          "E.g. 30% now, 30% on dip to S1, 40% on deeper dip to S2. "
          "Reduces timing risk and lowers average cost in declining markets.")
    gloss("Stop Loss",
          "Pre-defined price at which you exit a losing position to limit losses. "
          "Typically set below the second support level (S2) as a safety net. "
          "Rule of thumb: never risk more than 5–7% of capital per trade.")
    gloss("Risk/Reward Ratio",
          "Expected upside (to analyst target or DCF) ÷ Expected downside (to stop loss). "
          "A ratio of 2x means you expect to gain $2 for every $1 you risk. "
          "Most professional traders require at least 2:1 before entering a trade.")
    gloss("Composite Weight",
          "The relative importance assigned to each valuation method when computing Composite Fair Value. "
          "Methods returning N/A (e.g. DDM for non-dividend stocks) are automatically excluded and "
          "the remaining weights are re-normalised to 100%.")

    st.markdown("---")
    st.markdown(
        '<div style="color:#475569;font-size:0.75rem">All definitions are educational summaries. '
        'This app is not a substitute for professional financial advice.</div>',
        unsafe_allow_html=True
    )
