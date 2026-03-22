"""
Plotly chart builders for technical analysis
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.calculations import (
    sma, ema, bollinger_bands, rsi, macd, atr, find_support_resistance
)

DARK_BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#f1f5f9"
GRID_COLOR = "#334155"
BULLISH = "#22c55e"
BEARISH = "#ef4444"
NEUTRAL = "#eab308"
BLUE = "#3b82f6"
PURPLE = "#a855f7"
ORANGE = "#f97316"


def layout_defaults(title=""):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14)),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, size=11),
        xaxis=dict(
            gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
            tickfont=dict(color=TEXT_COLOR)
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
            tickfont=dict(color=TEXT_COLOR)
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COLOR, borderwidth=1,
            font=dict(color=TEXT_COLOR)
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )


def price_chart(df: pd.DataFrame, ticker: str, period: str = "1Y") -> go.Figure:
    """Main price chart with SMA/EMA, Bollinger Bands, Volume, Support/Resistance"""
    if df is None or df.empty:
        return go.Figure()

    close = df["Close"]
    dates = df.index

    sma50 = sma(close, 50)
    sma200 = sma(close, 200) if len(close) >= 200 else sma(close, min(len(close), 200))
    ema9 = ema(close, 9)
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    resistance, support = find_support_resistance(df)

    # Create subplots: price (top 70%) + volume (bottom 30%)
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{ticker} — Price & Indicators", "Volume"]
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=dates, open=df["Open"], high=df["High"],
        low=df["Low"], close=close,
        name="Price",
        increasing_line_color=BULLISH,
        decreasing_line_color=BEARISH,
        increasing_fillcolor=BULLISH,
        decreasing_fillcolor=BEARISH,
        showlegend=False,
    ), row=1, col=1)

    # ── Bollinger Bands (fill between upper/lower) ──
    fig.add_trace(go.Scatter(
        x=dates, y=bb_upper, name="BB Upper",
        line=dict(color="rgba(59,130,246,0.4)", width=1, dash="dot"),
        showlegend=True
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=bb_lower, name="BB Lower",
        fill="tonexty", fillcolor="rgba(59,130,246,0.05)",
        line=dict(color="rgba(59,130,246,0.4)", width=1, dash="dot"),
        showlegend=True
    ), row=1, col=1)

    # ── Moving Averages ──
    fig.add_trace(go.Scatter(
        x=dates, y=ema9, name="EMA 9",
        line=dict(color="#f97316", width=1.2)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=sma50, name="SMA 50",
        line=dict(color="#a855f7", width=1.5)
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=sma200, name="SMA 200",
        line=dict(color="#06b6d4", width=1.5)
    ), row=1, col=1)

    # ── Support & Resistance ──
    for i, level in enumerate(resistance[:3]):
        fig.add_hline(
            y=level, row=1, col=1,
            line=dict(color=BEARISH, width=1, dash="dash"),
            annotation_text=f"R{i+1} ${level:.2f}",
            annotation_font=dict(color=BEARISH, size=10)
        )
    for i, level in enumerate(support[:3]):
        fig.add_hline(
            y=level, row=1, col=1,
            line=dict(color=BULLISH, width=1, dash="dash"),
            annotation_text=f"S{i+1} ${level:.2f}",
            annotation_font=dict(color=BULLISH, size=10)
        )

    # ── Volume ──
    colors = [BULLISH if c >= o else BEARISH for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=dates, y=df["Volume"], name="Volume",
        marker_color=colors, opacity=0.7, showlegend=False
    ), row=2, col=1)

    fig.update_layout(**layout_defaults())
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=520,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    close = df["Close"]
    rsi_vals = rsi(close, 14)

    fig = go.Figure()

    # Color zones
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.08)", line_width=0)

    fig.add_trace(go.Scatter(
        x=df.index, y=rsi_vals, name="RSI (14)",
        line=dict(color=BLUE, width=1.5)
    ))
    fig.add_hline(y=70, line=dict(color=BEARISH, width=1, dash="dot"),
                  annotation_text="Overbought 70", annotation_font=dict(color=BEARISH, size=10))
    fig.add_hline(y=30, line=dict(color=BULLISH, width=1, dash="dot"),
                  annotation_text="Oversold 30", annotation_font=dict(color=BULLISH, size=10))
    fig.add_hline(y=50, line=dict(color=GRID_COLOR, width=1, dash="dot"))

    fig.update_layout(**layout_defaults("RSI (14-day)"))
    fig.update_layout(height=250, yaxis=dict(range=[0, 100]))
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    close = df["Close"]
    macd_line, signal_line, histogram = macd(close)
    histogram = histogram.fillna(0)  # guard NaN before color list comprehension

    hist_colors = [BULLISH if v >= 0 else BEARISH for v in histogram]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=histogram, name="Histogram",
        marker_color=hist_colors, opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=macd_line, name="MACD",
        line=dict(color=BLUE, width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=signal_line, name="Signal",
        line=dict(color=ORANGE, width=1.5)
    ))
    fig.add_hline(y=0, line=dict(color=GRID_COLOR, width=1))

    fig.update_layout(**layout_defaults("MACD (12, 26, 9)"))
    fig.update_layout(height=250)
    return fig


def bb_width_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    close = df["Close"]
    upper, mid, lower = bollinger_bands(close)
    # guard divide-by-zero: replace 0 mid values with NaN before dividing
    mid_safe = mid.replace(0, float("nan"))
    bb_width = (upper - lower) / mid_safe * 100  # as % of price

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_width, name="BB Width %",
        line=dict(color=PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(168,85,247,0.1)"
    ))
    fig.update_layout(**layout_defaults("Bollinger Band Width (Volatility)"))
    fig.update_layout(height=220)
    fig.update_yaxes(title_text="BB Width %")
    return fig


def atr_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return go.Figure()
    atr_vals = atr(df, 14)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=atr_vals, name="ATR (14)",
        line=dict(color=ORANGE, width=1.5),
        fill="tozeroy", fillcolor="rgba(249,115,22,0.1)"
    ))
    fig.update_layout(**layout_defaults("Average True Range (14-day)"))
    fig.update_layout(height=220)
    fig.update_yaxes(title_text="ATR ($)")
    return fig


def gauge_chart(score: int, title: str, low_color: str = BEARISH,
                high_color: str = BULLISH) -> go.Figure:
    """Circular gauge for score 0-100"""
    color = BULLISH if score >= 60 else (BEARISH if score < 40 else NEUTRAL)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"color": TEXT_COLOR, "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": TEXT_COLOR,
                     "tickfont": {"color": TEXT_COLOR}},
            "bar": {"color": color},
            "bgcolor": CARD_BG,
            "bordercolor": GRID_COLOR,
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                {"range": [40, 60], "color": "rgba(234,179,8,0.15)"},
                {"range": [60, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
        number={"font": {"color": color, "size": 32}, "suffix": "%"},
    ))
    fig.update_layout(
        paper_bgcolor=CARD_BG, font=dict(color=TEXT_COLOR),
        height=200, margin=dict(l=20, r=20, t=30, b=10)
    )
    return fig
