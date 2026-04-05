"""
Financial calculations: DCF, Graham Number, RSI, MACD, Bollinger Bands, etc.
"""
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# VALUATION METHODS
# ─────────────────────────────────────────────

def graham_number(eps, book_value_per_share):
    """√(22.5 × EPS × BVPS) — Benjamin Graham's intrinsic value formula."""
    try:
        if eps is None or book_value_per_share is None:
            return None
        if eps <= 0 or book_value_per_share <= 0:
            return None
        return round((22.5 * eps * book_value_per_share) ** 0.5, 2)
    except Exception:
        return None


def dcf_valuation(free_cash_flow, growth_rate, shares_outstanding,
                  beta=1.0, risk_free_rate=0.045, equity_risk_premium=0.055,
                  high_growth_years=5, fade_years=5, terminal_growth=0.03):
    """
    Two-stage DCF with CAPM-derived discount rate. Institutional grade.

    Stage 1: 'growth_rate' applied for high_growth_years.
    Stage 2: Growth linearly fades from growth_rate → terminal_growth over fade_years.
    Terminal value: Gordon Growth Model on the final projected FCF.

    Discount rate = risk_free_rate + beta × equity_risk_premium (CAPM).
    Default inputs reflect 2024/2025 US market: 10yr UST 4.5%, ERP 5.5%.
    A beta of 1.0 → WACC ≈ 10%; beta 1.3 (e.g. AMZN) → WACC ≈ 11.65%.

    10-year horizon gives a far more realistic terminal value capture than
    a 5-year model for compounding growth businesses.
    """
    try:
        if free_cash_flow is None or shares_outstanding is None:
            return None
        if free_cash_flow <= 0 or shares_outstanding <= 0:
            return None
        if growth_rate is None:
            growth_rate = 0.10

        # CAPM discount rate — min 3% above terminal growth to avoid division issues
        discount_rate = risk_free_rate + (beta or 1.0) * equity_risk_premium
        discount_rate = max(discount_rate, terminal_growth + 0.03)

        total_years = high_growth_years + fade_years
        pv = 0.0
        fcf_current = free_cash_flow

        for i in range(1, total_years + 1):
            if i <= high_growth_years:
                g = growth_rate
            else:
                # Linear fade: t=1 → still near growth_rate, t=fade_years → terminal_growth
                t = i - high_growth_years
                g = growth_rate - (growth_rate - terminal_growth) * (t / fade_years)

            fcf_current = fcf_current * (1 + g)
            pv += fcf_current / (1 + discount_rate) ** i

        # Terminal value on the last projected FCF
        terminal_value = fcf_current * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** total_years

        intrinsic_value = (pv + pv_terminal) / shares_outstanding
        return round(intrinsic_value, 2)
    except Exception:
        return None


def pe_based_valuation(eps, forward_eps, sector_pe):
    """
    EPS × sector benchmark P/E.

    Institutional preference: Forward EPS first (analyst consensus is forward-looking),
    then trailing EPS. Sector P/E from Damodaran/Bloomberg sector median.
    Using the stock's own P/E would always reproduce the current price (circular math).

    Returns: (value_or_None, eps_label_or_None)
    """
    try:
        # Prefer forward EPS — analysts price on forward earnings, not trailing
        if forward_eps is not None and forward_eps > 0:
            return round(forward_eps * sector_pe, 2), "fwd EPS"
        # Fall back to trailing EPS
        if eps is not None and eps > 0:
            return round(eps * sector_pe, 2), "trailing EPS"
        # Negative / unavailable EPS — PE-based valuation doesn't apply
        return None, None
    except Exception:
        return None, None


def ev_ebit_valuation(ebit, net_debt, shares_outstanding, ev_ebit_multiple):
    """
    EV/EBIT-based valuation — preferred institutional complement to EV/EBITDA.

    EV/EBIT strips out D&A, making it more conservative than EV/EBITDA and better
    for comparing capital-intensive vs asset-light companies on equal footing.
    Used heavily by PE firms and sell-side for LBO and cross-sector comparisons.

    Formula: Fair Price = (EBIT × multiple − Net Debt) / Shares Outstanding
    """
    try:
        if ebit is None or ebit <= 0 or shares_outstanding is None or shares_outstanding <= 0:
            return None
        enterprise_value = ebit * ev_ebit_multiple
        net_debt = net_debt or 0
        equity_value = enterprise_value - net_debt
        return round(max(equity_value / shares_outstanding, 0), 2)
    except Exception:
        return None


def fcf_yield_valuation(free_cash_flow, shares_outstanding, required_fcf_yield=0.04):
    """
    FCF Yield-based intrinsic value — anchors equity value to required investor return.

    Institutional logic: Fair Price = FCF per share / Required FCF Yield.
    Required yield = risk-free rate + equity risk premium for this asset class.
    - Growth (3–4%): accept low FCF yield for high future growth (tech, cloud)
    - Balanced (4–5%): mid-quality compounder
    - Conservative (5–6%): mature / value-oriented; demands higher yield

    Limitation: uses trailing FCF. High-capex companies (Amazon AWS, Intel fabs)
    have suppressed trailing FCF. This conservatively understates their earnings power.
    Balance against EV/EBITDA and DCF methods for a full picture.
    """
    try:
        if free_cash_flow is None or shares_outstanding is None:
            return None
        if free_cash_flow <= 0 or shares_outstanding <= 0:
            return None
        if required_fcf_yield <= 0:
            return None
        fcf_per_share = free_cash_flow / shares_outstanding
        return round(fcf_per_share / required_fcf_yield, 2)
    except Exception:
        return None


def ev_ebitda_valuation(ebitda, net_debt, shares_outstanding, ev_multiple):
    """
    EV/EBITDA-based valuation.
    ev_multiple: sector median EV/EBITDA (live or fallback from data.fetch_sector_multiples).
    """
    try:
        if ebitda is None or ebitda <= 0 or shares_outstanding is None or shares_outstanding <= 0:
            return None
        enterprise_value = ebitda * ev_multiple
        net_debt = net_debt or 0
        equity_value = enterprise_value - net_debt
        return round(max(equity_value / shares_outstanding, 0), 2)
    except Exception:
        return None


def pb_valuation(book_value_per_share, pb_multiple):
    """
    Book Value × sector P/B multiple.
    pb_multiple: sector median P/B (live or fallback from data.fetch_sector_multiples).
    """
    try:
        if book_value_per_share is None or book_value_per_share <= 0:
            return None
        return round(book_value_per_share * pb_multiple, 2)
    except Exception:
        return None


def peg_signal(pe_ratio, earnings_growth_rate):
    """PEG = P/E ÷ Earnings Growth Rate (%). <1 undervalued, >1.5 overvalued."""
    try:
        if pe_ratio is None or earnings_growth_rate is None:
            return None, None
        if earnings_growth_rate <= 0 or pe_ratio <= 0:
            return None, None
        peg = pe_ratio / (earnings_growth_rate * 100)
        signal = "Undervalued" if peg < 1 else ("Fairly Valued" if peg < 1.5 else "Overvalued")
        return round(peg, 2), signal
    except Exception:
        return None, None


def dividend_discount_model(dividend_per_share, dividend_growth_rate, discount_rate=0.10):
    """Gordon Growth Model: Intrinsic Value = D / (r - g)"""
    try:
        if dividend_per_share is None or dividend_per_share <= 0:
            return None
        if dividend_growth_rate is None:
            dividend_growth_rate = 0.03
        if discount_rate <= dividend_growth_rate:
            return None
        return round(dividend_per_share / (discount_rate - dividend_growth_rate), 2)
    except Exception:
        return None


def composite_fair_value(values: dict, weights: dict):
    """
    Weighted average of valid (non-None, positive) fair value estimates.

    values:  {method_key: value_or_None}   e.g. {"graham": 120.5, "dcf": None, ...}
    weights: {method_key: weight_0_to_100}  e.g. {"graham": 15, "dcf": 30, ...}

    Missing methods are silently skipped; remaining weights are auto-normalised.
    Returns None if no valid method is available.
    """
    total_weight = 0.0
    weighted_sum = 0.0
    for key, val in values.items():
        w = weights.get(key, 0)
        if val is not None and val > 0 and w > 0:
            weighted_sum += val * w
            total_weight += w
    if total_weight == 0:
        return None
    return round(weighted_sum / total_weight, 2)


def fundamental_signal(current_price, fair_value):
    """Return (signal_text, upside_pct)."""
    try:
        if current_price is None or fair_value is None or fair_value == 0:
            return "N/A", 0
        upside_pct = ((fair_value - current_price) / current_price) * 100
        if upside_pct > 15:
            return "🟢 Undervalued", upside_pct
        elif upside_pct < -15:
            return "🔴 Overvalued", upside_pct
        else:
            return "🟡 Fair Value", upside_pct
    except Exception:
        return "N/A", 0


def fundamental_score(current_price, fair_value, metrics: dict):
    """0-100 score based on valuation + quality metrics."""
    score = 50  # Base

    # Valuation component (up to ±25)
    if current_price and fair_value and fair_value > 0:
        upside = (fair_value - current_price) / current_price
        score += min(25, max(-25, upside * 50))

    roe = metrics.get("roe")
    if roe:
        if roe > 0.20:   score += 5
        elif roe > 0.12: score += 2
        elif roe < 0:    score -= 5

    net_margin = metrics.get("net_margin")
    if net_margin:
        if net_margin > 0.15:   score += 5
        elif net_margin > 0.08: score += 2
        elif net_margin < 0:    score -= 5

    de_ratio = metrics.get("debt_equity")
    if de_ratio is not None:
        if de_ratio < 0.5: score += 5
        elif de_ratio > 2: score -= 5

    current_ratio = metrics.get("current_ratio")
    if current_ratio:
        if current_ratio > 1.5: score += 3
        elif current_ratio < 1: score -= 3

    rev_growth = metrics.get("revenue_growth")
    if rev_growth:
        if rev_growth > 0.10:   score += 5
        elif rev_growth > 0:    score += 2
        elif rev_growth < -0.05: score -= 3

    short_pct = metrics.get("short_pct")
    if short_pct:
        if short_pct > 20: score -= 5
        elif short_pct > 10: score -= 2

    return max(0, min(100, round(score)))


# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(window=period, min_periods=1).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    # When avg_loss == 0 (all gains), RSI = 100 — avoids divide-by-zero → NaN
    rsi_val = np.where(
        avg_loss == 0,
        100.0,
        100.0 - (100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan)))
    )
    return pd.Series(rsi_val, index=series.index)


def macd(series: pd.Series, fast=12, slow=26, signal_period=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high = df["High"]
    low  = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def find_support_resistance(df: pd.DataFrame, window=5, n_levels=3):
    """Find swing highs (resistance) and swing lows (support)."""
    highs = df["High"].values
    lows  = df["Low"].values

    resistance = []
    support    = []

    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
            resistance.append(highs[i])
        if all(lows[i] <= lows[i - j] for j in range(1, window + 1)) and \
           all(lows[i] <= lows[i + j] for j in range(1, window + 1)):
            support.append(lows[i])

    def cluster_levels(levels, ascending=False):
        if not levels:
            return []
        sorted_l = sorted(set(levels), reverse=True)
        clusters = []
        used = set()
        for l in sorted_l:
            if l in used:
                continue
            group = [x for x in sorted_l if l != 0 and abs(x - l) / abs(l) < 0.01]
            if not group:
                group = [l]
            clusters.append(round(float(np.mean(group)), 2))
            used.update(group)
        result = clusters[:n_levels]
        result.sort(reverse=not ascending)
        return result

    # Resistance: ascending (nearest = lowest above price first)
    # Support:    descending (nearest = highest below price first)
    return cluster_levels(resistance, ascending=True), cluster_levels(support, ascending=False)


def technical_score(df: pd.DataFrame):
    """0-100 technical score from multiple signals."""
    if df is None or len(df) < 50:
        return 50, {}

    close   = df["Close"]
    current = close.iloc[-1]

    sma50_val  = sma(close, 50).iloc[-1]
    sma200_val = sma(close, 200).iloc[-1] if len(close) >= 200 else sma50_val
    ema9_val   = ema(close, 9).iloc[-1]
    rsi_val    = rsi(close, 14).iloc[-1]
    macd_line, signal_line, _ = macd(close)
    macd_val   = macd_line.iloc[-1]
    signal_val = signal_line.iloc[-1]

    def _safe(v, fallback=0.0):
        return fallback if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    sma50_val  = _safe(sma50_val,  current)
    sma200_val = _safe(sma200_val, current)
    ema9_val   = _safe(ema9_val,   current)
    rsi_val    = _safe(rsi_val,    50.0)
    macd_val   = _safe(macd_val,   0.0)
    signal_val = _safe(signal_val, 0.0)
    sma50, sma200, ema9 = sma50_val, sma200_val, ema9_val

    resistance, support = find_support_resistance(df)

    score   = 50
    signals = {}

    if current > sma200:
        score += 12;  signals["Price > SMA200"] = "✅ Bullish"
    else:
        score -= 8;   signals["Price > SMA200"] = "❌ Bearish"

    if current > sma50:
        score += 8;   signals["Price > SMA50"] = "✅ Bullish"
    else:
        score -= 5;   signals["Price > SMA50"] = "❌ Bearish"

    if ema9 > sma50:
        score += 5;   signals["EMA9 > SMA50"] = "✅ Bullish"
    else:
        score -= 3;   signals["EMA9 > SMA50"] = "❌ Bearish"

    if rsi_val < 30:
        score += 10;  signals[f"RSI ({rsi_val:.0f})"] = "✅ Oversold – Bullish"
    elif rsi_val > 70:
        score -= 10;  signals[f"RSI ({rsi_val:.0f})"] = "❌ Overbought – Bearish"
    elif 40 <= rsi_val <= 65:
        score += 5;   signals[f"RSI ({rsi_val:.0f})"] = "✅ Neutral-Bullish"
    else:
        signals[f"RSI ({rsi_val:.0f})"] = "⚪ Neutral"

    if macd_val > signal_val:
        score += 8;   signals["MACD"] = "✅ Bullish Crossover"
    else:
        score -= 5;   signals["MACD"] = "❌ Bearish Crossover"

    if support:
        nearest_support = min(support, key=lambda x: abs(x - current))
        if abs(current - nearest_support) / current < 0.03:
            score += 5
            signals["Support Level"] = f"✅ Near support ${nearest_support:.2f}"

    if resistance:
        nearest_resistance = min(resistance, key=lambda x: abs(x - current))
        if abs(current - nearest_resistance) / current < 0.03:
            score -= 5
            signals["Resistance Level"] = f"❌ Near resistance ${nearest_resistance:.2f}"

    return max(0, min(100, round(score))), signals


# ─────────────────────────────────────────────
# COMBINED DECISION
# ─────────────────────────────────────────────

def investment_decision(fund_score: int, tech_score: int):
    """Combine fundamental (60%) + technical (40%) into BUY/SELL/HOLD."""
    combined = fund_score * 0.6 + tech_score * 0.4
    if combined >= 70:
        signal, action = "🟢 BUY",       "BUY"
    elif combined >= 57:
        signal, action = "🟩 ACCUMULATE", "ACCUMULATE"
    elif combined >= 43:
        signal, action = "🟡 HOLD",       "HOLD"
    elif combined >= 30:
        signal, action = "🟠 REDUCE",     "REDUCE"
    else:
        signal, action = "🔴 SELL",       "SELL"
    return signal, action, round(combined)


def tranche_plan(action, current_price, support_levels, resistance_levels):
    """
    5-tranche deployment plan — each tranche = 20% of allocated capital.

    DOWNSIDE TABLE (buy into weakness):
      5 entry levels spread from current price to ~35% below.
      Max drawdown cap: ~38% below T1 (stop sits 3% under T5).
      Price levels snap to nearby chart supports where available.

    UPSIDE TABLE (scale-out / profit targets):
      5 exit levels spread from ~8% above to ~40% above current price.
      Price levels snap to nearby chart resistances where available.

    Returns
    -------
    down_plan  : list[dict] — 5 downside rows + 1 summary row
    up_plan    : list[dict] — 5 upside rows + 1 summary row
    stop_loss  : float — hard stop (3% below T5 downside level)
    levels     : dict — d1..d5, u1..u5 price floats for metric cards
    """
    cp = current_price  # shorthand

    # ── Default downside levels: 0%, -8%, -16%, -25%, -35% ────────────────
    _d_defaults = [0.00, 0.08, 0.16, 0.25, 0.35]

    # ── Snap to chart supports where a support falls within ±2.5% of default ─
    below_s = sorted(
        [s for s in support_levels if s < cp * 0.998],
        reverse=True   # nearest first
    )

    def _snap_down(default_pct, used):
        """Return a downside price level, snapping to nearest unused support."""
        target = cp * (1 - default_pct)
        for s in below_s:
            if s in used:
                continue
            if abs(s - target) / cp <= 0.025:   # within 2.5% of default
                used.add(s)
                return round(s, 2)
        return round(target, 2)

    used_d = set()
    d_raw = [_snap_down(pct, used_d) for pct in _d_defaults]

    # Enforce strict descending order with at least 5% separation
    d = [round(cp, 2)]
    for i in range(1, 5):
        min_allowed = round(d[i - 1] * (1 - 0.05), 2)
        candidate   = d_raw[i]
        d.append(min(candidate, min_allowed))

    d1, d2, d3, d4, d5 = d
    stop_loss = round(d5 * 0.97, 2)   # 3% below T5 ≈ 37-38% from entry

    # ── Default upside levels: +8%, +15%, +23%, +30%, +40% ────────────────
    _u_defaults = [0.08, 0.15, 0.23, 0.30, 0.40]

    above_r = sorted(
        [r for r in resistance_levels if r > cp * 1.002]
    )

    def _snap_up(default_pct, used):
        target = cp * (1 + default_pct)
        for r in above_r:
            if r in used:
                continue
            if abs(r - target) / cp <= 0.025:
                used.add(r)
                return round(r, 2)
        return round(target, 2)

    used_u = set()
    u_raw = [_snap_up(pct, used_u) for pct in _u_defaults]

    # Enforce strict ascending order with at least 5% separation
    u = [round(u_raw[0], 2)]
    for i in range(1, 5):
        min_allowed = round(u[i - 1] * 1.05, 2)
        candidate   = u_raw[i]
        u.append(max(candidate, min_allowed))

    u1, u2, u3, u4, u5 = u

    # ── Compute % distances ────────────────────────────────────────────────
    def _pct_dn(price): return round((cp - price) / cp * 100, 1)
    def _pct_up(price): return round((price - cp) / cp * 100, 1)

    # ── Weighted average cost (all 5 tranches equal weight) ────────────────
    avg_cost_down = round(sum([d1, d2, d3, d4, d5]) / 5, 2)
    avg_cost_up   = round(sum([u1, u2, u3, u4, u5]) / 5, 2)

    # ── Downside plan rows ─────────────────────────────────────────────────
    down_labels = [
        "T1 · 20% — Initiate",
        "T2 · 20% — Add",
        "T3 · 20% — Add more",
        "T4 · 20% — Heavy add",
        "T5 · 20% — Max load",
    ]
    down_rationale = [
        "First entry — undervalued signal confirmed",
        f"Pullback to -{_pct_dn(d2):.1f}% — reduce avg cost",
        f"Deeper support at -{_pct_dn(d3):.1f}% — scale in",
        f"Strong support zone at -{_pct_dn(d4):.1f}% — largest add",
        f"Max drawdown level at -{_pct_dn(d5):.1f}% — final tranche",
    ]
    down_prices  = [d1, d2, d3, d4, d5]

    down_plan = []
    for i, (lbl, price, rat) in enumerate(zip(down_labels, down_prices, down_rationale)):
        pct_str = "—" if i == 0 else f"-{_pct_dn(price):.1f}%"
        down_plan.append({
            "Tranche":    lbl,
            "Price":      f"${price:,.2f}",
            "Δ From Now": pct_str,
            "Rationale":  rat,
        })
    # Summary row
    sl_pct = round((cp - stop_loss) / cp * 100, 1)
    ac_pct = round((cp - avg_cost_down) / cp * 100, 1)
    down_plan.append({
        "Tranche":    "📊 SUMMARY",
        "Price":      f"Avg Cost: ${avg_cost_down:,.2f}",
        "Δ From Now": f"-{ac_pct:.1f}% avg",
        "Rationale":  f"Stop Loss ${stop_loss:,.2f} (-{sl_pct:.1f}%) · Max drawdown ~{sl_pct:.0f}% on T1",
    })

    # ── Upside plan rows ───────────────────────────────────────────────────
    up_labels = [
        "T1 · Trim 20% — Quick profit",
        "T2 · Trim 20% — R1 resistance",
        "T3 · Trim 20% — Mid target",
        "T4 · Trim 20% — Extended move",
        "T5 · Exit 20% — Full target",
    ]
    up_rationale = [
        f"+{_pct_up(u1):.1f}% — lock in first tranche, let rest run",
        f"+{_pct_up(u2):.1f}% — first resistance cluster, book gains",
        f"+{_pct_up(u3):.1f}% — mid-run exit, reduces position risk",
        f"+{_pct_up(u4):.1f}% — extended move; only strong conviction holds",
        f"+{_pct_up(u5):.1f}% — full target; close remaining position",
    ]
    up_prices = [u1, u2, u3, u4, u5]

    up_plan = []
    for lbl, price, rat in zip(up_labels, up_prices, up_rationale):
        up_plan.append({
            "Tranche":    lbl,
            "Price":      f"${price:,.2f}",
            "Δ From Now": f"+{_pct_up(price):.1f}%",
            "Rationale":  rat,
        })
    # Summary row
    avg_exit_pct = round((avg_cost_up - cp) / cp * 100, 1)
    up_plan.append({
        "Tranche":    "📊 SUMMARY",
        "Price":      f"Avg Exit: ${avg_cost_up:,.2f}",
        "Δ From Now": f"+{avg_exit_pct:.1f}% avg",
        "Rationale":  f"Blended exit if all 5 targets hit · R:R vs stop = {round(avg_exit_pct/sl_pct,2):.2f}×",
    })

    levels = {
        "d1": d1, "d2": d2, "d3": d3, "d4": d4, "d5": d5,
        "u1": u1, "u2": u2, "u3": u3, "u4": u4, "u5": u5,
        "avg_cost_down": avg_cost_down,
        "avg_cost_up":   avg_cost_up,
    }

    return down_plan, up_plan, stop_loss, levels


# ─────────────────────────────────────────────
# UNIVERSAL CONFLUENCE ENGINE (UCE)
# ─────────────────────────────────────────────

class StockDecisionEngine:
    """
    Universal Confluence Engine — synthesizes 4 technical pillars into a
    single confidence score and actionable trade recommendation.

    Pillars (max score):
      1. Momentum — RSI-14 + 5-day slope          (25 pts)
      2. Trend    — MACD 12,26,9 crossover + hist  (35 pts)
      3. Volatility — BB Width squeeze + ATR stop  (25 pts)
      4. Relative Strength — 30d alpha vs SPY      (15 pts)
    Total: 100 pts

    Thresholds:
      > 80  → High-Confidence Buy
      50–80 → Speculative Buy
      < 50  → Neutral / Avoid
    """

    def __init__(self, ticker: str, df: pd.DataFrame,
                 spy_df: pd.DataFrame = None,
                 trade_type: str = "Swing",
                 max_drawdown_pct: float = 5.0):
        self.ticker          = ticker
        self.df              = (df.copy() if isinstance(df, pd.DataFrame) and not df.empty
                                else pd.DataFrame())
        self.spy_df          = spy_df.copy() if spy_df is not None and not spy_df.empty else None
        self.trade_type      = trade_type      # "Swing" | "Day"
        self.max_drawdown_pct = max_drawdown_pct

        # Swing vs Day: ATR multiplier and RSI thresholds differ
        if trade_type == "Day":
            self._atr_mult          = 1.5   # tighter stop for intraday
            self._rsi_oversold      = 25    # stricter oversold
            self._rsi_overbought    = 75
            self._rs_lookback       = 10    # 2-week alpha window
        else:
            self._atr_mult          = 2.5
            self._rsi_oversold      = 30
            self._rsi_overbought    = 70
            self._rs_lookback       = 30    # 30-day alpha window

    # ─── Public entry point ───────────────────────────────────────
    def analyze(self) -> dict:
        if self.df.empty or len(self.df) < 30:
            return {"error": "Insufficient data — need at least 30 trading days."}

        close = self.df["Close"]
        current_price = float(close.iloc[-1])

        p1 = self._pillar1_momentum(close)
        p2 = self._pillar2_trend(close)
        p3 = self._pillar3_volatility(close, current_price)
        p4 = self._pillar4_relative_strength(close)

        # Raw score capped 0-100
        raw_score = p1["score"] + p2["score"] + p3["score"] + p4["score"]
        confidence = max(0, min(100, raw_score))

        # Recommendation
        if confidence > 80:
            recommendation = "🟢 High-Confidence Buy"
            action_label   = "HIGH-CONFIDENCE BUY"
            rec_color      = "#14532d"
            rec_border     = "#22c55e"
        elif confidence >= 50:
            recommendation = "🟡 Speculative Buy"
            action_label   = "SPECULATIVE BUY"
            rec_color      = "#713f12"
            rec_border     = "#eab308"
        else:
            recommendation = "⚪ Neutral / Avoid"
            action_label   = "AVOID"
            rec_color      = "#1e293b"
            rec_border     = "#475569"

        # ── Stop Loss Logic ───────────────────────────────────────
        smart_stop       = p3["smart_stop"]             # 2.5×ATR below price
        drawdown_stop    = current_price * (1 - self.max_drawdown_pct / 100)
        # Effective stop: whichever is higher (less loss)
        effective_stop   = max(smart_stop, drawdown_stop)
        stop_pct         = (current_price - effective_stop) / current_price * 100
        stop_driver      = ("Max-Drawdown Rule" if drawdown_stop > smart_stop
                            else f"Smart Stop ({self._atr_mult}×ATR)")

        # ── Position Sizing ───────────────────────────────────────
        risk_per_share   = current_price - effective_stop
        # Shares needed so that a stop-out costs exactly max_drawdown_pct of $10k
        if risk_per_share > 0:
            shares_per_10k = int(10_000 * (self.max_drawdown_pct / 100) / risk_per_share)
        else:
            shares_per_10k = 0
        capital_at_risk_10k = round(risk_per_share * shares_per_10k, 2)

        # Profit target: reward-to-risk multiple
        # Swing = 2.5×R, Day = 2.0×R — institutional minimums
        reward_multiple = 2.5 if self.trade_type == "Swing" else 2.0
        target_price    = round(current_price + risk_per_share * reward_multiple, 2)
        target_pct      = round((target_price - current_price) / current_price * 100, 2)

        return {
            "ticker":           self.ticker,
            "current_price":    current_price,
            "trade_type":       self.trade_type,
            "max_drawdown_pct": self.max_drawdown_pct,
            "confidence":       round(confidence, 1),
            "recommendation":   recommendation,
            "action_label":     action_label,
            "rec_color":        rec_color,
            "rec_border":       rec_border,
            "pillars": {
                "momentum":      p1,
                "trend":         p2,
                "volatility":    p3,
                "rel_strength":  p4,
            },
            "entry_price":          round(current_price, 2),
            "smart_stop":           round(smart_stop, 2),
            "effective_stop":       round(effective_stop, 2),
            "stop_pct":             round(stop_pct, 2),
            "stop_driver":          stop_driver,
            "risk_per_share":       round(risk_per_share, 2),
            "shares_per_10k":       shares_per_10k,
            "capital_at_risk_10k":  capital_at_risk_10k,
            "target_price":         target_price,
            "target_pct":           target_pct,
            "reward_multiple":      reward_multiple,
        }

    # ─── Pillar 1: Momentum (RSI-14) ─────────────────────────────
    def _pillar1_momentum(self, close: pd.Series) -> dict:
        rsi_series  = rsi(close, 14)
        curr_rsi    = float(rsi_series.iloc[-1])

        # 5-day RSI slope (momentum shift detector)
        lookback    = min(5, len(rsi_series) - 1)
        slope       = float(rsi_series.iloc[-1] - rsi_series.iloc[-1 - lookback])

        # Score based on RSI zone
        if curr_rsi < self._rsi_oversold:
            signal = f"🟢 Extreme Value — Oversold (RSI < {self._rsi_oversold})"
            score  = 25
        elif curr_rsi < 40:
            signal = "🟢 Approaching Oversold"
            score  = 18
        elif curr_rsi < 50:
            signal = "🟡 Below Midline — Mild Bearish Momentum"
            score  = 10
        elif curr_rsi < 60:
            signal = "🟡 Above Midline — Mild Bullish Momentum"
            score  = 13
        elif curr_rsi < self._rsi_overbought:
            signal = f"🟠 Approaching Overbought (RSI < {self._rsi_overbought})"
            score  = 8
        else:
            signal = f"🔴 Overextension — Overbought (RSI > {self._rsi_overbought})"
            score  = 0

        # Slope bonus: RSI below 50 AND rising → early momentum shift
        if slope > 3 and curr_rsi < 50:
            slope_signal = "↑ Rising — Early Momentum Shift ✅"
            score = min(score + 5, 25)
        elif slope > 1:
            slope_signal = "↗ Gradually Rising"
        elif slope < -3:
            slope_signal = "↓ Falling — Momentum Weakening"
            if curr_rsi > 50:
                score = max(score - 3, 0)
        elif slope < -1:
            slope_signal = "↘ Gradually Falling"
        else:
            slope_signal = "→ Flat"

        return {
            "rsi":            round(curr_rsi, 2),
            "rsi_slope_5d":   round(slope, 2),
            "signal":         signal,
            "slope_signal":   slope_signal,
            "score":          score,
            "max_score":      25,
        }

    # ─── Pillar 2: Trend (MACD 12,26,9) ──────────────────────────
    def _pillar2_trend(self, close: pd.Series) -> dict:
        if len(close) < 26:
            return {"signal": "⚪ Insufficient data for MACD", "score": 0,
                    "max_score": 35, "macd": None, "signal_line": None,
                    "histogram": None, "below_zero": None, "histogram_signal": "—"}

        macd_line, signal_line, histogram = macd(close)

        curr_macd   = float(macd_line.iloc[-1])
        curr_sig    = float(signal_line.iloc[-1])
        curr_hist   = float(histogram.iloc[-1])
        prev_macd   = float(macd_line.iloc[-2])
        prev_sig    = float(signal_line.iloc[-2])
        prev_hist   = float(histogram.iloc[-2])

        below_zero  = curr_macd < 0
        cross_now   = curr_macd > curr_sig and prev_macd <= prev_sig   # fresh bullish cross
        bear_cross  = curr_macd < curr_sig and prev_macd >= prev_sig   # fresh bearish cross

        # Check for crossover within last 3 bars
        recent_bull_cross = False
        for i in range(-4, -1):
            if (abs(i) < len(macd_line) and
                    float(macd_line.iloc[i]) > float(signal_line.iloc[i]) and
                    float(macd_line.iloc[i - 1]) <= float(signal_line.iloc[i - 1])):
                recent_bull_cross = True
                break

        hist_expanding = abs(curr_hist) > abs(prev_hist)

        # Scoring
        if cross_now and below_zero:
            signal = "🟢 Bullish Crossover Below Zero — Strongest Signal"
            score  = 35
        elif cross_now and not below_zero:
            signal = "🟢 Bullish Crossover Above Zero"
            score  = 22
        elif recent_bull_cross and below_zero:
            signal = "🟢 Recent Bullish Cross Below Zero (within 3 bars)"
            score  = 28
        elif recent_bull_cross:
            signal = "🟡 Recent Bullish Cross (above zero)"
            score  = 18
        elif curr_macd > curr_sig:
            signal = "🟡 MACD Above Signal — No Fresh Crossover"
            score  = 12
        elif bear_cross:
            signal = "🔴 Fresh Bearish Crossover"
            score  = 0
        else:
            signal = "🔴 MACD Below Signal — Bearish Trend"
            score  = 5

        # Histogram modifier (±5, within [0, max_score])
        if hist_expanding and score >= 20:
            hist_signal = "📈 Expanding — Trend Accelerating ✅"
            score = min(score + 5, 35)
        elif not hist_expanding and score >= 20:
            hist_signal = "📉 Contracting — Trend Exhaustion ⚠️"
            score = max(score - 5, 0)
        elif hist_expanding:
            hist_signal = "📈 Expanding"
        else:
            hist_signal = "📉 Contracting"

        return {
            "macd":             round(curr_macd, 4),
            "signal_line":      round(curr_sig, 4),
            "histogram":        round(curr_hist, 4),
            "below_zero":       below_zero,
            "signal":           signal,
            "histogram_signal": hist_signal,
            "score":            score,
            "max_score":        35,
        }

    # ─── Pillar 3: Volatility (BB Width + ATR) ───────────────────
    def _pillar3_volatility(self, close: pd.Series, current_price: float) -> dict:
        upper, mid, lower = bollinger_bands(close, period=20, std_dev=2)
        bb_width = ((upper - lower) / mid.replace(0, np.nan)).fillna(0)

        curr_bbw = float(bb_width.iloc[-1])
        lookback = min(60, len(bb_width))
        bbw_60d  = bb_width.iloc[-lookback:]
        pct20    = float(bbw_60d.quantile(0.20))
        squeeze  = curr_bbw <= pct20
        bbw_pct  = float((bbw_60d <= curr_bbw).mean() * 100)

        # ATR smart stop
        atr_val     = atr(self.df, period=14)
        curr_atr    = float(atr_val.iloc[-1])
        smart_stop  = current_price - self._atr_mult * curr_atr
        stop_pct    = (current_price - smart_stop) / current_price * 100

        # Scoring
        if squeeze:
            signal = f"🟢 Volatility Squeeze — BB Width ≤ 20th percentile ({bbw_pct:.0f}th). Breakout Imminent"
            score  = 25
        elif bbw_pct <= 40:
            signal = f"🟡 Low-Normal Volatility (BB Width at {bbw_pct:.0f}th percentile)"
            score  = 15
        elif bbw_pct <= 70:
            signal = f"🟡 Normal Volatility (BB Width at {bbw_pct:.0f}th percentile)"
            score  = 8
        else:
            signal = f"🔴 Elevated Volatility — Risk Raised (BB Width at {bbw_pct:.0f}th percentile)"
            score  = 0

        return {
            "bb_width":         round(curr_bbw, 5),
            "bb_pct_rank":      round(bbw_pct, 1),
            "squeeze_active":   squeeze,
            "atr_14":           round(curr_atr, 3),
            "atr_mult":         self._atr_mult,
            "smart_stop":       round(smart_stop, 2),
            "smart_stop_pct":   round(stop_pct, 2),
            "signal":           signal,
            "score":            score,
            "max_score":        25,
        }

    # ─── Pillar 4: Relative Strength vs SPY ──────────────────────
    def _pillar4_relative_strength(self, close: pd.Series) -> dict:
        lb = self._rs_lookback
        if len(close) < lb:
            return {"signal": "⚪ Insufficient data", "score": 5, "max_score": 15,
                    "ticker_return": None, "spy_return": None, "alpha": None}

        ticker_ret = float((close.iloc[-1] / close.iloc[-lb] - 1) * 100)

        spy_ret = None
        if self.spy_df is not None and len(self.spy_df) >= lb:
            sc = self.spy_df["Close"]
            spy_ret = float((sc.iloc[-1] / sc.iloc[-lb] - 1) * 100)

        alpha = (ticker_ret - spy_ret) if spy_ret is not None else None

        if alpha is not None:
            if alpha > 10:
                signal = f"🟢 Strong Outperformer — Alpha +{alpha:.1f}% vs SPY"
                score  = 15
            elif alpha > 5:
                signal = f"🟢 Outperforming SPY by +{alpha:.1f}%"
                score  = 12
            elif alpha > 0:
                signal = f"🟡 Slight Outperformance (+{alpha:.1f}% alpha)"
                score  = 8
            elif alpha > -5:
                signal = f"🟠 Slight Underperformance ({alpha:.1f}% alpha)"
                score  = 4
            else:
                signal = f"🔴 Underperforming SPY by {alpha:.1f}%"
                score  = 0
        else:
            # Fallback: absolute return only
            if ticker_ret > 5:
                signal = f"🟡 +{ticker_ret:.1f}% return ({lb}d) — No SPY data"
                score  = 8
            elif ticker_ret > 0:
                signal = f"🟡 +{ticker_ret:.1f}% ({lb}d) — No SPY data"
                score  = 5
            else:
                signal = f"🟠 {ticker_ret:.1f}% ({lb}d) — No SPY data"
                score  = 2

        return {
            "ticker_return":  round(ticker_ret, 2),
            "spy_return":     round(spy_ret, 2) if spy_ret is not None else None,
            "alpha":          round(alpha, 2) if alpha is not None else None,
            "lookback_days":  lb,
            "signal":         signal,
            "score":          score,
            "max_score":      15,
        }
