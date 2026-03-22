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
                  discount_rate=0.10, terminal_growth=0.03, years=5):
    """
    5-year DCF with terminal value.
    Returns per-share intrinsic value.
    """
    try:
        if free_cash_flow is None or shares_outstanding is None:
            return None
        if free_cash_flow <= 0 or shares_outstanding <= 0:
            return None
        if growth_rate is None:
            growth_rate = 0.08
        if discount_rate <= terminal_growth:
            return None

        pv = 0.0
        for i in range(1, years + 1):
            fcf_i = free_cash_flow * (1 + growth_rate) ** i
            pv += fcf_i / (1 + discount_rate) ** i

        terminal_fcf = free_cash_flow * (1 + growth_rate) ** years * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / (1 + discount_rate) ** years

        intrinsic_value = (pv + pv_terminal) / shares_outstanding
        return round(intrinsic_value, 2)
    except Exception:
        return None


def pe_based_valuation(eps, forward_eps, sector_pe):
    """
    EPS × sector benchmark P/E.

    Uses trailing EPS first; falls back to forward EPS for growth / loss-making companies.
    sector_pe should come from live sector ETF data (see data.fetch_sector_multiples)
    or industry-consensus fallback (labeled accordingly in the UI).

    Returns: (value_or_None, eps_label_or_None)
    """
    try:
        # Prefer trailing EPS (actual earnings)
        if eps is not None and eps > 0:
            return round(eps * sector_pe, 2), "trailing EPS"
        # Fall back to forward EPS (analyst estimates) for growth / pre-profit companies
        if forward_eps is not None and forward_eps > 0:
            return round(forward_eps * sector_pe, 2), "fwd EPS"
        # Negative / unavailable EPS — PE-based valuation doesn't apply
        return None, None
    except Exception:
        return None, None


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
    """Generate tranche deployment / exit plan."""
    s1 = support_levels[0]    if len(support_levels) > 0 else current_price * 0.92
    s2 = support_levels[1]    if len(support_levels) > 1 else current_price * 0.85
    r1 = resistance_levels[0] if len(resistance_levels) > 0 else current_price * 1.08
    r2 = resistance_levels[1] if len(resistance_levels) > 1 else current_price * 1.15

    stop_loss = round(min(s2 * 0.97, current_price * 0.85), 2)

    if action in ("BUY", "ACCUMULATE"):
        plan = [
            {"Tranche": "1st Buy", "% Capital": "30%", "Price Level": f"${current_price:.2f}",
             "Condition": "Enter now", "Rationale": "At fair/undervalued level"},
            {"Tranche": "2nd Buy", "% Capital": "30%", "Price Level": f"${s1:.2f}",
             "Condition": f"Price drops to ${s1:.2f}", "Rationale": "Add at Support 1"},
            {"Tranche": "3rd Buy", "% Capital": "40%", "Price Level": f"${s2:.2f}",
             "Condition": f"Price drops to ${s2:.2f}", "Rationale": "Full position at deep value"},
        ]
    elif action == "HOLD":
        plan = [
            {"Tranche": "No new buy", "% Capital": "—", "Price Level": "—",
             "Condition": "Monitor", "Rationale": "Wait for breakout or pullback"},
            {"Tranche": "Trim 20%",   "% Capital": "20%", "Price Level": f"${r1:.2f}",
             "Condition": f"Price reaches ${r1:.2f}", "Rationale": "Book partial profits at R1"},
        ]
    else:  # REDUCE / SELL
        plan = [
            {"Tranche": "Trim 30%", "% Capital": "30%", "Price Level": f"${r1:.2f}",
             "Condition": f"Price at/above ${r1:.2f}", "Rationale": "Reduce at Resistance 1"},
            {"Tranche": "Trim 40%", "% Capital": "40%", "Price Level": f"${r2:.2f}",
             "Condition": f"Price at/above ${r2:.2f}", "Rationale": "Further reduction at R2"},
            {"Tranche": "Exit 30%", "% Capital": "30%", "Price Level": f"${stop_loss:.2f}",
             "Condition": f"Stop loss hit ${stop_loss:.2f}", "Rationale": "Full exit on stop"},
        ]

    return plan, stop_loss, s1, s2, r1, r2
