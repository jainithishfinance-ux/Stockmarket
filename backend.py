"""
StockVision Pro — Python Backend (Flask)
Run: pip install flask flask-cors yfinance anthropic
Then: python backend.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np
import json
import os
from datetime import datetime, timedelta

# ──────────────────────────────────────────
#  CONFIG  (set your keys as env variables)
# ──────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("AIzaSyBUzhfYkXW-Epk4imiQPRVcYkVR1azijMo")
# export ANTHROPIC_API_KEY=sk-ant-xxxx

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────

def safe_float(val, fallback=0.0):
    try:
        v = float(val)
        return v if not (v != v) else fallback  # NaN check
    except Exception:
        return fallback


# ──────────────────────────────────────────
#  TECHNICAL INDICATORS
# ──────────────────────────────────────────

def compute_rsi(closes, period=14):
    closes = np.array(closes, dtype=float)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    rsi_vals = [None] * (period + 1)
    if avg_loss == 0:
        rsi_vals.append(100.0)
    else:
        rsi_vals.append(100 - 100 / (1 + avg_gain / avg_loss))
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_vals.append(100.0)
        else:
            rsi_vals.append(100 - 100 / (1 + avg_gain / avg_loss))
    return rsi_vals


def compute_ema(closes, period):
    closes = np.array(closes, dtype=float)
    k = 2 / (period + 1)
    ema = [closes[0]]
    for price in closes[1:]:
        ema.append(price * k + ema[-1] * (1 - k))
    return ema


def compute_macd(closes):
    ema12 = compute_ema(closes, 12)
    ema26 = compute_ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = compute_ema(macd_line, 9)
    hist = [a - b for a, b in zip(macd_line, signal)]
    return macd_line, signal, hist


def compute_bollinger(closes, period=20, k=2):
    closes = np.array(closes, dtype=float)
    upper, mid, lower = [], [], []
    for i in range(len(closes)):
        if i < period - 1:
            upper.append(None); mid.append(None); lower.append(None)
        else:
            window = closes[i - period + 1:i + 1]
            m = np.mean(window); s = np.std(window)
            mid.append(float(m)); upper.append(float(m + k * s)); lower.append(float(m - k * s))
    return upper, mid, lower


def compute_sma(closes, period):
    closes = np.array(closes, dtype=float)
    result = []
    for i in range(len(closes)):
        if i < period - 1:
            result.append(None)
        else:
            result.append(float(np.mean(closes[i - period + 1:i + 1])))
    return result


def compute_atr(ohlc, period=14):
    tr_list = []
    for i, bar in enumerate(ohlc):
        if i == 0:
            tr_list.append(bar['high'] - bar['low'])
        else:
            prev_close = ohlc[i - 1]['close']
            tr = max(bar['high'] - bar['low'],
                     abs(bar['high'] - prev_close),
                     abs(bar['low'] - prev_close))
            tr_list.append(tr)
    return compute_sma(tr_list, period)


# ──────────────────────────────────────────
#  ML — SIMPLE MODELS (pure numpy)
# ──────────────────────────────────────────

def build_features(ohlc):
    closes = [d['close'] for d in ohlc]
    volumes = [d['volume'] for d in ohlc]
    rsi_vals = compute_rsi(closes)
    sma20 = compute_sma(closes, 20)
    sma50 = compute_sma(closes, min(50, len(closes)))
    macd_line, macd_sig, macd_hist = compute_macd(closes)
    bb_upper, bb_mid, bb_lower = compute_bollinger(closes)
    atr_vals = compute_atr(ohlc)

    c_arr = np.array(closes); v_arr = np.array(volumes, dtype=float)
    c_norm = (c_arr - c_arr.min()) / (c_arr.max() - c_arr.min() + 1e-9)
    v_norm = (v_arr - v_arr.min()) / (v_arr.max() - v_arr.min() + 1e-9)

    features = []
    for i, d in enumerate(ohlc):
        c = d['close']; o = d['open']
        prev_c = ohlc[i - 1]['close'] if i > 0 else c
        ret1 = (c - prev_c) / prev_c
        ret5 = (c - ohlc[i - 5]['close']) / ohlc[i - 5]['close'] if i >= 5 else 0
        ret20 = (c - ohlc[i - 20]['close']) / ohlc[i - 20]['close'] if i >= 20 else 0
        rsi = safe_float(rsi_vals[i], 50)
        ml = safe_float(macd_line[i], 0)
        ms = safe_float(macd_sig[i], 0)
        mh = safe_float(macd_hist[i], 0)
        s20d = ((c - sma20[i]) / sma20[i]) if sma20[i] else 0
        s50d = ((c - sma50[i]) / sma50[i]) if sma50[i] else 0
        bu = bb_upper[i]; bl = bb_lower[i]
        bbpos = (c - bl) / (bu - bl) if bu and bl and (bu - bl) > 0 else 0.5
        atr_pct = (atr_vals[i] / c) if atr_vals[i] and c else 0
        features.append([
            rsi / 100, (ml + 10) / 20, mh, s20d, s50d,
            bbpos, atr_pct * 10, ret1 * 10, ret5 * 5, ret20 * 3,
            ((c - o) / o) * 10, ((d['high'] - d['low']) / d['low']) * 10,
            float(c_norm[i]), float(v_norm[i])
        ])
    return features


def train_linear_regression(X, y, epochs=300, lr=0.01):
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, m = X.shape
    w = np.zeros(m); b = 0.0
    for _ in range(epochs):
        pred = X @ w + b
        err = pred - y
        w -= lr * (X.T @ err) / n
        b -= lr * np.mean(err)
    return w, b


def predict_lr(X, w, b):
    return (np.array(X, dtype=float) @ w + b).tolist()


def run_ml_models(ohlc):
    closes = [d['close'] for d in ohlc]
    features = build_features(ohlc)
    win = 10
    X, y_reg, y_cls = [], [], []
    for i in range(win, len(features) - 1):
        X.append(features[i])
        ret = (closes[i + 1] - closes[i]) / closes[i]
        y_reg.append(ret)
        y_cls.append(1 if closes[i + 1] > closes[i] else 0)

    results = {"lstm": {}, "rf": {}, "lr": {}, "ensemble": "HOLD", "ensConf": "50",
               "currentPrice": closes[-1], "technicals": {}}

    if len(X) < 10:
        return results

    # Linear Regression (continuous)
    w_reg, b_reg = train_linear_regression(X, y_reg)
    lr_pred = predict_lr([features[-1]], w_reg, b_reg)[0]
    lr_price = closes[-1] * (1 + lr_pred)

    # Logistic-style (binary)
    w_cls, b_cls = train_linear_regression(X, y_cls)
    lr_cls = predict_lr([features[-1]], w_cls, b_cls)[0]

    # Simple momentum model
    last_returns = [y_reg[-1], y_reg[-2] if len(y_reg) > 1 else 0, y_reg[-3] if len(y_reg) > 2 else 0]
    momentum_pred = np.mean(last_returns)

    # Ensemble
    bull_votes = sum([lr_pred > 0, lr_cls > 0.5, momentum_pred > 0])
    ensemble = "BUY" if bull_votes >= 2 else "SELL"
    ens_conf = round((bull_votes / 3) * 100)

    # Last technicals
    feat = features[-1]
    results["lstm"] = {
        "predictedReturn": round(lr_pred * 100, 3),
        "predictedPrice": round(lr_price, 2),
        "direction": "UP" if lr_pred > 0 else "DOWN"
    }
    results["rf"] = {
        "direction": "UP" if lr_cls > 0.5 else "DOWN",
        "confidence": str(round(abs(lr_cls - 0.5) * 100 + 50, 1))
    }
    results["lr"] = {
        "score": str(round(lr_cls, 3)),
        "direction": "UP" if lr_cls > 0.5 else "DOWN"
    }
    results["ensemble"] = ensemble
    results["ensConf"] = str(ens_conf)
    results["currentPrice"] = closes[-1]
    results["technicals"] = {
        "rsi": safe_float(feat[0] * 100),
        "macdHist": safe_float(feat[2]),
        "sma20Dist": safe_float(feat[3]),
        "sma50Dist": safe_float(feat[4]),
        "bbPos": safe_float(feat[5]),
        "atrPct": safe_float(feat[6] / 10)
    }
    return results


# ──────────────────────────────────────────
#  ROUTES
# ──────────────────────────────────────────

@app.route("/api/quote/<symbol>")
def get_quote(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2d")
        price = safe_float(info.get("regularMarketPrice") or (hist['Close'].iloc[-1] if len(hist) else 0))
        prev = safe_float(info.get("previousClose") or (hist['Close'].iloc[-2] if len(hist) > 1 else price))
        change = price - prev
        return jsonify({
            "symbol": symbol.upper(),
            "price": price,
            "open": safe_float(info.get("open", price)),
            "high": safe_float(info.get("dayHigh", price)),
            "low": safe_float(info.get("dayLow", price)),
            "prevClose": prev,
            "volume": int(info.get("regularMarketVolume", 0) or 0),
            "change": change,
            "changePct": (change / prev * 100) if prev else 0,
            "mktCap": safe_float(info.get("marketCap", 0)),
            "52wHigh": safe_float(info.get("fiftyTwoWeekHigh", 0)),
            "52wLow": safe_float(info.get("fiftyTwoWeekLow", 0)),
            "pe": safe_float(info.get("trailingPE", 0)),
            "beta": safe_float(info.get("beta", 1)),
        })
    except Exception as e:
        return jsonify({"error": str(e), "symbol": symbol, "price": 0, "change": 0, "changePct": 0}), 200


@app.route("/api/ohlc/<symbol>")
def get_ohlc(symbol):
    period = request.args.get("period", "6mo")
    interval = request.args.get("interval", "1d")
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval=interval)
        rows = []
        for dt, row in hist.iterrows():
            rows.append({
                "date": dt.strftime("%Y-%m-%d"),
                "open": safe_float(row['Open']),
                "high": safe_float(row['High']),
                "low": safe_float(row['Low']),
                "close": safe_float(row['Close']),
                "volume": int(row['Volume'])
            })
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/indicators/<symbol>")
def get_indicators(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="6mo")
        closes = hist['Close'].tolist()
        ohlc = [{"open": r['Open'], "high": r['High'], "low": r['Low'],
                 "close": r['Close'], "volume": r['Volume']}
                for _, r in hist.iterrows()]
        rsi_vals = compute_rsi(closes)
        macd_line, macd_sig, macd_hist = compute_macd(closes)
        sma20 = compute_sma(closes, 20)
        sma50 = compute_sma(closes, 50)
        bb_upper, bb_mid, bb_lower = compute_bollinger(closes)
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        return jsonify({
            "dates": dates, "closes": closes,
            "rsi": rsi_vals, "macd_line": macd_line,
            "macd_signal": macd_sig, "macd_hist": macd_hist,
            "sma20": sma20, "sma50": sma50,
            "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/<symbol>")
def get_ml(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="6mo")
        ohlc = [{"open": float(r['Open']), "high": float(r['High']),
                 "low": float(r['Low']), "close": float(r['Close']),
                 "volume": float(r['Volume'])}
                for _, r in hist.iterrows()]
        result = run_ml_models(ohlc)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-predict", methods=["POST"])
def ai_predict():
    """Call Claude API for AI-enhanced prediction"""
    try:
        import anthropic
    except ImportError:
        return jsonify({"error": "anthropic not installed. Run: pip install anthropic"}), 500

    if not ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 400

    data = request.json
    sym = data.get("symbol", "AAPL")
    ml = data.get("ml", {})
    closes = data.get("closes", [])

    n = len(closes)
    if n < 21:
        return jsonify({"error": "Not enough price data"}), 400

    last10 = ", ".join([str(round(c, 2)) for c in closes[-10:]])
    chg5 = round((closes[-1] - closes[-6]) / closes[-6] * 100, 2) if n >= 6 else 0
    chg20 = round((closes[-1] - closes[-21]) / closes[-21] * 100, 2) if n >= 21 else 0

    prompt = f"""You are a quantitative analyst. Analyze {sym} and provide a structured prediction.

MARKET DATA:
• Price: ${round(closes[-1], 2)}
• Last 10 closes: {last10}
• 5-day return: {chg5}%
• 20-day return: {chg20}%
• RSI(14): {ml.get('technicals', {}).get('rsi', 50):.1f}
• MACD Hist: {ml.get('technicals', {}).get('macdHist', 0):.4f}
• BB Position: {ml.get('technicals', {}).get('bbPos', 0.5)*100:.1f}%

ML ENSEMBLE: {ml.get('ensemble','HOLD')} ({ml.get('ensConf','50')}% agreement)

Respond ONLY with valid JSON, no markdown:
{{
  "verdict": "BUY"|"SELL"|"HOLD",
  "confidence": 0-100,
  "target_1d": price,
  "target_1w": price,
  "target_1m": price,
  "direction_1d": "UP"|"DOWN",
  "direction_1w": "UP"|"DOWN",
  "prob_up_1d": 0-100,
  "prob_up_1w": 0-100,
  "risk_level": "LOW"|"MEDIUM"|"HIGH",
  "support": price,
  "resistance": price,
  "reasoning": "2-3 sentence analysis"
}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        text = msg.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return jsonify(json.loads(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news/<symbol>")
def get_news(symbol):
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        result = []
        for n in news[:8]:
            ct = n.get("content", {})
            result.append({
                "headline": ct.get("title") or n.get("title", ""),
                "summary": (ct.get("summary") or ct.get("description") or "")[:200],
                "source": (ct.get("provider", {}) or {}).get("displayName", "Yahoo Finance"),
                "url": ct.get("canonicalUrl", {}).get("url") if ct.get("canonicalUrl") else "#",
                "datetime": ct.get("pubDate", "")
            })
        return jsonify(result)
    except Exception as e:
        return jsonify([])


@app.route("/api/search/<query>")
def search_stocks(query):
    """Quick symbol search"""
    COMMON = [
        {"s": "AAPL", "n": "Apple Inc."}, {"s": "MSFT", "n": "Microsoft"},
        {"s": "GOOGL", "n": "Alphabet"}, {"s": "AMZN", "n": "Amazon"},
        {"s": "NVDA", "n": "NVIDIA"}, {"s": "TSLA", "n": "Tesla"},
        {"s": "META", "n": "Meta Platforms"}, {"s": "NFLX", "n": "Netflix"},
        {"s": "AMD", "n": "AMD"}, {"s": "JPM", "n": "JPMorgan"},
        {"s": "V", "n": "Visa"}, {"s": "MA", "n": "Mastercard"},
        {"s": "DIS", "n": "Disney"}, {"s": "PYPL", "n": "PayPal"},
        {"s": "CRM", "n": "Salesforce"}, {"s": "ORCL", "n": "Oracle"},
        {"s": "UBER", "n": "Uber"}, {"s": "SHOP", "n": "Shopify"},
        {"s": "COIN", "n": "Coinbase"}, {"s": "PLTR", "n": "Palantir"},
        {"s": "INTC", "n": "Intel"}, {"s": "GS", "n": "Goldman Sachs"},
        {"s": "BAC", "n": "Bank of America"}, {"s": "IBM", "n": "IBM"},
        {"s": "SNAP", "n": "Snap"}, {"s": "SPOT", "n": "Spotify"},
        {"s": "SQ", "n": "Block"}, {"s": "LYFT", "n": "Lyft"},
        {"s": "RIVN", "n": "Rivian"}, {"s": "NIO", "n": "NIO"},
    ]
    q = query.upper()
    matches = [s for s in COMMON if q in s["s"] or q in s["n"].upper()]
    return jsonify(matches[:6])


@app.route("/api/batch-quotes", methods=["POST"])
def batch_quotes():
    symbols = request.json.get("symbols", [])
    results = {}
    for sym in symbols[:15]:  # limit
        try:
            t = yf.Ticker(sym)
            info = t.info
            hist = t.history(period="2d")
            price = safe_float(info.get("regularMarketPrice") or (hist['Close'].iloc[-1] if len(hist) else 0))
            prev = safe_float(info.get("previousClose") or (hist['Close'].iloc[-2] if len(hist) > 1 else price))
            change = price - prev
            results[sym] = {
                "price": price, "change": change,
                "changePct": (change / prev * 100) if prev else 0
            }
        except Exception:
            results[sym] = {"price": 0, "change": 0, "changePct": 0}
    return jsonify(results)


@app.route("/")
def index():
    return jsonify({"status": "StockVision Pro API running", "version": "3.0"})


if __name__ == "__main__":
    print("=" * 50)
    print("  StockVision Pro — Python Backend")
    print("  Running on http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host="0.0.0.0")
