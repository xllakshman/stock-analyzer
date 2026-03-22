"""
Stock Analyzer — Streamlit App
Covers NASDAQ 100, S&P 100, NIFTY 100
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils.tickers import UNIVERSE_MAP
from utils.data import fetch_history, extract_fundamentals
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
# CUSTOM CSS — Bloomberg-light dark theme
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0f172a; color: #f1f5f9; }
  [data-testid="stSidebar"] { background-color: #1e293b; }
  [data-testid="stSidebar"] .stMarkdown { color: #f1f5f9; }

  /* Metric cards */
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

  /* Signal badges */
  .badge-buy    { background:#15803d; color:#dcfce7; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-accum  { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-hold   { background:#854d0e; color:#fef9c3; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-reduce { background:#9a3412; color:#ffedd5; padding:3px 10px; border-radius:4px; font-weight:600; }
  .badge-sell   { background:#7f1d1d; color:#fee2e2; padding:3px 10px; border-radius:4px; font-weight:600; }

  /* Header bar */
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

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { background: #1e293b; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"]      { color: #94a3b8; border-radius: 6px; font-weight: 500; }
  .stTabs [aria-selected="true"]    { background: #3b82f6 !important; color: #fff !important; }

  /* Tables */
  .stDataFrame { background: #1e293b !important; }

  /* Inputs */
  .stTextInput input, .stSelectbox select { background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
  .stNumberInput input { background: #1e293b !important; color: #f1f5f9 !important; }

  /* Divider */
  hr { border-color: #334155; }

  /* Progress bar */
  .progress-container { background:#334155; border-radius:4px; height:8px; overflow:hidden; }
  .progress-fill      { height:8px; border-radius:4px; }
  .stProgress > div > div { background-color: #3b82f6 !important; }

  /* Risk flag */
  .risk-flag { background:#7f1d1d; color:#fca5a5; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
  .ok-flag   { background:#14532d; color:#86efac; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.85rem; }
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


def valuation_card(label, value, current_price):
    if value is None:
        diff_html = '<span style="color:#64748b">N/A</span>'
        val_str = "N/A"
    else:
        val_str = f"${value:,.2f}"
        if current_price:
            diff_pct = (value - current_price) / current_price * 100
            color = "#22c55e" if diff_pct > 5 else ("#ef4444" if diff_pct < -5 else "#eab308")
            arrow = "▲" if diff_pct > 0 else "▼"
            diff_html = f'<span style="color:{color}">{arrow} {abs(diff_pct):.1f}% {"upside" if diff_pct>0 else "downside"}</span>'
        else:
            diff_html = ""
    st.markdown(f"""
    <div class="metric-card" style="border-left:3px solid #3b82f6;">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{val_str}</div>
      <div class="metric-sub">{diff_html}</div>
    </div>""", unsafe_allow_html=True)


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
        selected_name = ticker_input
    else:
        tickers_list = UNIVERSE_MAP[universe]
        options = [f"{sym} — {name}" for sym, name in tickers_list]
        selected_option = st.selectbox("Select Stock", options, key="stock_picker")
        idx = options.index(selected_option)
        selected_ticker = tickers_list[idx][0]
        selected_name = tickers_list[idx][1]

    st.markdown("---")
    period = st.radio("Chart Period", ["1W", "1M", "3M", "6M", "1Y", "2Y", "5Y"], index=4, horizontal=True)

    st.markdown("---")
    dcf_growth = st.slider("DCF Growth Rate (%)", 0, 30, 10, 1) / 100

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

# Header
current_price = fund_data.get("current_price")
name = fund_data.get("name", selected_name)

# Compute 1-day change from history
day_change = None
day_change_pct = None
if hist_df is not None and len(hist_df) >= 2:
    day_change = hist_df["Close"].iloc[-1] - hist_df["Close"].iloc[-2]
    day_change_pct = day_change / hist_df["Close"].iloc[-2] * 100

price_str = f"${current_price:,.2f}" if current_price else "N/A"
chg_str = (
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
    "📊 Fundamental Analysis",
    "📈 Technical Analysis",
    "🎯 Investment Decision",
    "🏦 Universe Browser",
])

# ══════════════════════════════════════════════════════════════
# TAB 1: FUNDAMENTAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab1:
    eps              = fund_data.get("eps")
    bvps             = fund_data.get("book_value")
    fcf              = fund_data.get("free_cash_flow")
    shares           = fund_data.get("shares_outstanding")
    pe               = fund_data.get("trailing_pe")
    ebitda           = fund_data.get("ebitda")
    net_debt         = fund_data.get("net_debt", 0)
    sector           = fund_data.get("sector", "default")
    div_rate         = fund_data.get("dividend_rate")
    div_growth       = fund_data.get("dividend_growth")
    earnings_growth  = fund_data.get("earnings_growth")
    peg              = fund_data.get("peg_ratio")

    # ── COMPUTE FAIR VALUES ──
    graham = graham_number(eps, bvps)
    dcf    = dcf_valuation(fcf, dcf_growth, shares)
    pe_val = pe_based_valuation(eps, sector)  # sector benchmark P/E, not stock's own (avoids circular math)
    ev_val = ev_ebitda_valuation(ebitda, net_debt, shares, sector)
    pb_val = pb_valuation(bvps, sector)
    # DDM: use 4% default dividend growth rate.
    # fiveYearAvgDividendYield is a yield %, not a growth rate — don't use it directly.
    # A conservative 3-5% perpetual growth is standard for mature dividend payers.
    ddm_growth = 0.04  # 4% default dividend growth rate
    ddm = dividend_discount_model(div_rate, ddm_growth)
    peg_v, peg_sig = peg_signal(pe, earnings_growth)

    composite = composite_fair_value([graham, dcf, pe_val, ev_val, pb_val, ddm])
    signal_text, upside_pct = fundamental_signal(current_price, composite)

    metrics_for_score = {
        "roe": fund_data.get("roe"),
        "net_margin": fund_data.get("profit_margin"),
        "debt_equity": fund_data.get("debt_equity"),
        "current_ratio": fund_data.get("current_ratio"),
        "revenue_growth": fund_data.get("revenue_growth"),
        "short_pct": (fund_data.get("short_pct") or 0) * 100,
    }
    f_score = fundamental_score(current_price, composite, metrics_for_score)

    # ── SIGNAL BANNER ──
    banner_color = "#14532d" if "Undervalued" in signal_text else ("#7f1d1d" if "Overvalued" in signal_text else "#713f12")
    border_color = "#22c55e" if "Undervalued" in signal_text else ("#ef4444" if "Overvalued" in signal_text else "#eab308")
    composite_str = f"${composite:,.2f}" if composite else "N/A"
    # Bug fix: use `is not None` — upside_pct of 0.0 is valid and falsy
    upside_str    = f"{upside_pct:+.1f}%" if upside_pct is not None else "N/A"

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
        valuation_card("Graham Number", graham, current_price)
    with c2:
        valuation_card(f"DCF (5-yr, {dcf_growth*100:.0f}% growth)", dcf, current_price)
    with c3:
        # Show the sector benchmark P/E being used (not stock's own P/E)
        SECTOR_PE_DISPLAY = {
            "Technology": 28, "Communication Services": 22,
            "Consumer Discretionary": 22, "Consumer Staples": 20,
            "Healthcare": 22, "Financials": 13, "Industrials": 20,
            "Energy": 12, "Materials": 15, "Real Estate": 35,
            "Utilities": 17, "default": 20,
        }
        benchmark_pe = SECTOR_PE_DISPLAY.get(sector, 20)
        valuation_card(f"PE-Based (sector avg {benchmark_pe}x)", pe_val, current_price)

    c4, c5, c6 = st.columns(3)
    with c4:
        valuation_card(f"EV/EBITDA ({sector})", ev_val, current_price)
    with c5:
        valuation_card("Price-to-Book", pb_val, current_price)
    with c6:
        if div_rate and div_rate > 0:
            valuation_card("Dividend Discount Model (4% growth)", ddm, current_price)
        else:
            st.markdown("""
            <div class="metric-card" style="border-left:3px solid #334155">
              <div class="metric-label">Dividend Discount Model</div>
              <div class="metric-value" style="color:#64748b">N/A</div>
              <div class="metric-sub">Not a dividend-paying stock</div>
            </div>""", unsafe_allow_html=True)

    # PEG — shown separately with an explanation note
    if peg_v is not None:
        peg_color = "#22c55e" if peg_v < 1 else ("#ef4444" if peg_v > 1.5 else "#eab308")
        own_pe_str = f" (stock P/E: {pe:.1f}x)" if pe else ""
        st.markdown(f"""
        <div class="metric-card" style="border-left:3px solid {peg_color}">
          <div class="metric-label">PEG Ratio{own_pe_str}</div>
          <div class="metric-value" style="color:{peg_color}">{peg_v:.2f} — {peg_sig}</div>
          <div class="metric-sub">
            PEG = P/E ÷ Earnings Growth Rate. &lt;1 = growth justifies valuation, &gt;1.5 = expensive relative to growth.<br>
            <span style="color:#64748b;font-size:0.72rem">
              ⚠️ PEG and Composite Fair Value can disagree: PEG rewards fast-growing companies;
              absolute methods (Graham, DCF) may still flag them as expensive on a pure value basis.
              Both views are valid and complementary.
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── KEY METRICS ──
    st.markdown("### Key Fundamental Metrics")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**📈 Growth**")
        metric_card("Revenue Growth (YoY)", fmt_pct(fund_data.get("revenue_growth")))
        metric_card("Revenue 3Y CAGR",      fmt_pct(fund_data.get("revenue_3yr_cagr")))
        metric_card("Earnings Growth (YoY)", fmt_pct(fund_data.get("earnings_growth")))
        metric_card("Earnings 3Y CAGR",      fmt_pct(fund_data.get("earnings_3yr_cagr")))
        metric_card("Free Cash Flow",         fmt_currency(fcf),
                    f"Yield: {fmt_pct(fund_data.get('fcf_yield'))}")

    with col_b:
        st.markdown("**💰 Profitability & Quality**")
        metric_card("EBITDA Margin",     fmt_pct(fund_data.get("ebitda_margin")))
        metric_card("Net Profit Margin", fmt_pct(fund_data.get("profit_margin")))
        metric_card("Return on Equity (ROE)",   fmt_pct(fund_data.get("roe")))
        metric_card("Return on Assets (ROA)",   fmt_pct(fund_data.get("roa")))
        metric_card("ROCE (Approx.)",           fmt_pct(fund_data.get("roce")))

    with col_c:
        st.markdown("**⚖️ Risk & Ownership**")
        metric_card("Debt / Equity",       fmt_num(fund_data.get("debt_equity")))
        metric_card("Current Ratio",       fmt_num(fund_data.get("current_ratio")))
        metric_card("Interest Coverage",   fmt_num(fund_data.get("interest_coverage")))
        metric_card("Insider Ownership",   fmt_pct(fund_data.get("insider_ownership")))
        metric_card("Institutional Own.",  fmt_pct(fund_data.get("institutional_ownership")))

    col_d, col_e = st.columns(2)
    with col_d:
        st.markdown("**🎯 Analyst & Market**")
        metric_card("Short Interest %",  fmt_pct(fund_data.get("short_pct")),
                    "High >15% is bearish pressure")
        rec_raw = fund_data.get("recommendation", "") or ""
        rec = rec_raw.upper().replace("_", " ").strip() or "N/A"
        rec_colors = {
            "BUY": "#15803d", "STRONG BUY": "#14532d", "STRONGBUY": "#14532d",
            "HOLD": "#713f12", "NEUTRAL": "#713f12",
            "SELL": "#7f1d1d", "UNDERPERFORM": "#7f1d1d", "STRONG SELL": "#7f1d1d",
            "OVERWEIGHT": "#15803d", "UNDERWEIGHT": "#7f1d1d",
            "N/A": "#334155",
        }
        rec_color = rec_colors.get(rec.replace(" ","").upper(),
                    rec_colors.get(rec, "#334155"))
        st.markdown(f"""<div class="metric-card">
          <div class="metric-label">Analyst Recommendation</div>
          <div class="metric-value" style="color:{rec_color}">{rec}</div>
          <div class="metric-sub">Avg Target: {fmt_currency(fund_data.get('analyst_target'))} &nbsp;|&nbsp;
          Low: {fmt_currency(fund_data.get('analyst_low'))} &nbsp;|&nbsp;
          High: {fmt_currency(fund_data.get('analyst_high'))}</div>
        </div>""", unsafe_allow_html=True)
        metric_card("Beta (Market Risk)", fmt_num(fund_data.get("beta")),
                    "Beta >1 = more volatile than market")

    with col_e:
        st.markdown("**📊 52-Week Range**")
        lo = fund_data.get("fifty_two_week_low")
        hi = fund_data.get("fifty_two_week_high")
        if lo and hi and current_price:
            pos = (current_price - lo) / (hi - lo) if hi != lo else 0.5
            pos_pct = round(pos * 100, 1)
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">52-Week Range</div>
              <div style="display:flex;justify-content:space-between;color:#f1f5f9;font-size:0.9rem;margin-top:6px">
                <span>${lo:,.2f}</span>
                <span style="color:#3b82f6;font-weight:600">${current_price:,.2f} ({pos_pct}%)</span>
                <span>${hi:,.2f}</span>
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

# ══════════════════════════════════════════════════════════════
# TAB 2: TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    if hist_df is None or hist_df.empty:
        st.error(f"No historical data available for {selected_ticker}.")
    else:
        t_score, t_signals = technical_score(hist_df)

        # Signal banner
        sig_text = "🟢 Bullish" if t_score >= 60 else ("🔴 Bearish" if t_score < 40 else "🟡 Neutral")
        sig_color = "#14532d" if t_score >= 60 else ("#7f1d1d" if t_score < 40 else "#713f12")
        sig_border = "#22c55e" if t_score >= 60 else ("#ef4444" if t_score < 40 else "#eab308")

        st.markdown(f"""
        <div style="background:{sig_color};border:1px solid {sig_border};border-radius:8px;
        padding:12px 20px;margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
          <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{sig_text}</span>
          <span style="color:{sig_border};font-size:1.5rem;font-weight:700">Technical Score: {t_score}%</span>
        </div>""", unsafe_allow_html=True)

        # Signal breakdown
        with st.expander("📋 Signal Breakdown", expanded=False):
            cols = st.columns(2)
            items = list(t_signals.items())
            for i, (k, v) in enumerate(items):
                cols[i % 2].markdown(f"**{k}**: {v}")

        # Charts
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

    # Gauge charts
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

    # Main signal
    action_colors = {
        "BUY": "#15803d", "ACCUMULATE": "#166534", "HOLD": "#854d0e",
        "REDUCE": "#9a3412", "SELL": "#7f1d1d"
    }
    action_text_colors = {
        "BUY": "#dcfce7", "ACCUMULATE": "#bbf7d0", "HOLD": "#fef9c3",
        "REDUCE": "#ffedd5", "SELL": "#fee2e2"
    }
    bg = action_colors.get(action, "#334155")
    tc = action_text_colors.get(action, "#f1f5f9")

    st.markdown(f"""
    <div style="background:{bg};border-radius:12px;padding:24px;text-align:center;margin:16px 0">
      <div style="font-size:2.5rem;font-weight:800;color:{tc}">{signal_full}</div>
      <div style="color:{tc};opacity:0.8;font-size:1rem;margin-top:4px">
        Combined Score: {combined_score}% &nbsp;|&nbsp;
        Fundamental 60% + Technical 40%
      </div>
    </div>""", unsafe_allow_html=True)

    # Tranche Plan
    if current_price and hist_df is not None and not hist_df.empty:
        from utils.calculations import find_support_resistance
        resistance, support = find_support_resistance(hist_df)
        plan, stop_loss, s1, s2, r1, r2 = tranche_plan(action, current_price, support, resistance)

        st.markdown("### 📋 Tranche Deployment Plan")
        plan_df = pd.DataFrame(plan)
        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True,
        )

        # Risk metrics
        st.markdown("### ⚠️ Risk Metrics")
        r1c, r2c, r3c, r4c = st.columns(4)

        # downside_to_s2 is negative when s2 < current_price (correct: support below price)
        downside_to_s2 = ((s2 - current_price) / current_price * 100) if current_price else None
        analyst_target = fund_data.get("analyst_target")
        dcf_val = dcf
        upside_analyst = ((analyst_target - current_price) / current_price * 100) if analyst_target and current_price else None
        upside_dcf     = ((dcf_val - current_price) / current_price * 100)         if dcf_val     and current_price else None
        # Use best available upside; prefer analyst target
        best_upside = upside_analyst if upside_analyst is not None else upside_dcf
        # R/R = upside% / abs(downside%); both inputs must be non-zero
        rr_ratio = (
            round(best_upside / abs(downside_to_s2), 2)
            if best_upside is not None and downside_to_s2 and downside_to_s2 != 0
            else None
        )

        with r1c:
            # downside_to_s2 is expected to be negative — show as-is with sign
            ds = f"{downside_to_s2:.1f}%" if downside_to_s2 is not None else "N/A"
            metric_card("Downside to S2", ds, f"Strong support at ${s2:.2f}")
        with r2c:
            us = f"{upside_analyst:+.1f}%" if upside_analyst is not None else "N/A"
            metric_card("Upside to Analyst Target", us, fmt_currency(analyst_target))
        with r3c:
            us2 = f"{upside_dcf:+.1f}%" if upside_dcf is not None else "N/A"
            metric_card("Upside to DCF Value", us2, fmt_currency(dcf_val))
        with r4c:
            rr = f"{rr_ratio:.1f}x" if rr_ratio is not None else "N/A"
            rr_color = "#22c55e" if (rr_ratio is not None and rr_ratio >= 2) else ("#ef4444" if (rr_ratio is not None and rr_ratio < 1) else "#eab308")
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

    # Risk Flags
    st.markdown("### 🚩 Risk Flags")
    flags = []
    de = fund_data.get("debt_equity")
    cr = fund_data.get("current_ratio")
    si = fund_data.get("short_pct")
    margin = fund_data.get("profit_margin")
    roe_v = fund_data.get("roe")

    if de and de > 2:       flags.append(("risk", f"⚠️ High Debt/Equity: {de:.2f}x (>2 is elevated)"))
    elif de and de < 0.5:   flags.append(("ok",   f"✅ Low Debt/Equity: {de:.2f}x (strong balance sheet)"))
    if cr and cr < 1:       flags.append(("risk", f"⚠️ Low Current Ratio: {cr:.2f} (liquidity concern)"))
    elif cr and cr > 2:     flags.append(("ok",   f"✅ Strong Current Ratio: {cr:.2f}"))
    if si and si > 0.15:    flags.append(("risk", f"⚠️ High Short Interest: {si*100:.1f}% (bearish sentiment)"))
    if margin and margin < 0: flags.append(("risk", f"⚠️ Negative Net Margin: {margin*100:.1f}%"))
    elif margin and margin > 0.15: flags.append(("ok", f"✅ Strong Margin: {margin*100:.1f}%"))
    if roe_v and roe_v < 0: flags.append(("risk", f"⚠️ Negative ROE: {roe_v*100:.1f}%"))
    elif roe_v and roe_v > 0.15: flags.append(("ok", f"✅ Strong ROE: {roe_v*100:.1f}%"))

    if not flags:
        st.markdown('<div class="ok-flag">✅ No major risk flags detected</div>', unsafe_allow_html=True)
    for flag_type, msg in flags:
        css_class = "risk-flag" if flag_type == "risk" else "ok-flag"
        st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4: UNIVERSE BROWSER
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏦 Stock Universe Browser")
    st.markdown("Browse all stocks in an index with key metrics. Click any row to load its analysis.")

    uni_choice = st.selectbox(
        "Select Universe to Browse",
        ["NASDAQ 100", "S&P 100", "NIFTY 100"],
        key="universe_browse"
    )

    tickers_to_show = UNIVERSE_MAP[uni_choice]

    # Sector filter
    sectors = sorted(set(s for _, s in tickers_to_show))
    show_all = st.checkbox("Load all tickers (may take 2-3 minutes)", value=False)

    if show_all:
        with st.spinner(f"Fetching data for {len(tickers_to_show)} stocks..."):
            from utils.data import fetch_universe_snapshot
            df_universe = fetch_universe_snapshot(tickers_to_show)
    else:
        # Show static list with just names
        df_universe = pd.DataFrame([
            {"Ticker": sym, "Company": name}
            for sym, name in tickers_to_show
        ])
        st.info("Check 'Load all tickers' above to fetch live prices and metrics for all stocks (uses multiple API calls).")

    # Sortable table
    if "Change %" in df_universe.columns:
        df_styled = df_universe.style.format({
            "Price": "${:,.2f}",
            "Change %": "{:+.2f}%",
            "Market Cap ($B)": "${:.1f}B",
            "P/E": "{:.1f}",
        }).background_gradient(
            subset=["Change %"], cmap="RdYlGn", vmin=-5, vmax=5
        )
        st.dataframe(df_styled, use_container_width=True, height=500)
    else:
        st.dataframe(df_universe, use_container_width=True, height=500)

    st.markdown(f"**Total stocks in {uni_choice}:** {len(tickers_to_show)}")

    # Quick jump
    st.markdown("---")
    jump_options = [f"{sym} — {nm}" for sym, nm in tickers_to_show]
    jump_sel = st.selectbox("🔍 Quick-load a stock for analysis", ["— select —"] + jump_options)
    if jump_sel != "— select —":
        jump_ticker = jump_sel.split(" — ")[0]
        st.info(f"Select **{jump_ticker}** in the sidebar (or type it under Custom Ticker) to analyze it.")
