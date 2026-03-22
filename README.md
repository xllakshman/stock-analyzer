# 📈 Stock Analyzer — Streamlit App

Comprehensive stock analysis covering **NASDAQ 100**, **S&P 100**, and **NIFTY 100** with:
- 7 Fair Value methods (Graham Number, DCF, PE, EV/EBITDA, P/B, PEG, DDM)
- Full Technical Analysis (RSI, MACD, Bollinger Bands, ATR, Support/Resistance)
- Investment Decision engine (BUY/ACCUMULATE/HOLD/REDUCE/SELL)
- Tranche deployment plan & Risk metrics

**Data source**: Yahoo Finance (via `yfinance`) · No API key required · 15-min cache

---

## 🚀 Option A — Run Locally

### Prerequisites
- Python 3.9+
- pip

### Install & Run

**Mac / Linux:**
```bash
cd stock-analyzer-streamlit
pip install -r requirements.txt
streamlit run app.py
```

**Windows (Command Prompt):**
```cmd
cd stock-analyzer-streamlit
pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell):**
```powershell
cd stock-analyzer-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

### Virtual Environment (recommended)
```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Option B — Deploy to Streamlit Community Cloud (FREE, permanent URL)

### Step 1 — Push to GitHub

```bash
cd stock-analyzer-streamlit

# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit: Stock Analyzer"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/stock-analyzer.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Select:
   - **Repository**: `YOUR_USERNAME/stock-analyzer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

Your app will be live at:
```
https://YOUR_USERNAME-stock-analyzer-app-XXXX.streamlit.app
```

It auto-redeploys whenever you push to GitHub. **Completely free** for public repos.

---

## 📁 Project Structure

```
stock-analyzer-streamlit/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dark theme configuration
└── utils/
    ├── __init__.py
    ├── tickers.py            # NASDAQ 100, S&P 100, NIFTY 100 ticker lists
    ├── data.py               # yfinance data fetching + caching
    ├── calculations.py       # Financial math (DCF, Graham, RSI, MACD, etc.)
    └── charts.py             # Plotly chart builders
```

---

## 🎛️ How to Use

1. **Sidebar**: Choose a stock universe (NASDAQ 100 / S&P 100 / NIFTY 100) or type any custom ticker
2. **Fundamental Analysis tab**: See all 7 valuation methods, composite fair value, and 15+ key metrics
3. **Technical Analysis tab**: Interactive price chart with indicators, RSI, MACD, volatility charts
4. **Investment Decision tab**: BUY/HOLD/SELL signal, tranche entry plan, risk metrics
5. **Universe Browser tab**: Browse all stocks in an index (load live data optionally)

### NIFTY 100 Note
NIFTY 100 tickers use the `.NS` suffix (NSE India). The app handles this automatically — just select from the dropdown.

---

## ⚙️ Customization

- **DCF Growth Rate**: Editable slider in sidebar (0–30%)
- **Chart Period**: 1W / 1M / 3M / 6M / 1Y / 2Y / 5Y
- **Cache**: 15-minute TTL via `@st.cache_data(ttl=900)`. Click "🔍 Analyze" to force refresh.

---

## ⚠️ Disclaimer

This app is for **educational and research purposes only**. It is not financial advice. Always do your own due diligence before making any investment decisions.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Data shows N/A | Some metrics unavailable for certain stocks (e.g. NIFTY stocks may have limited coverage) |
| Rate limit errors | Wait 1-2 minutes and try again. Yahoo Finance has rate limits. |
| Streamlit Cloud deploy fails | Ensure `requirements.txt` is in the root directory |
| NIFTY stock not found | Try adding `.NS` to the ticker manually (e.g. `RELIANCE.NS`) |
