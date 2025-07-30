import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from babel.numbers import format_currency          # pip install babel

MODELS_DIR = Path("models")
SEQ_LEN = 60
HOST, PORT = "0.0.0.0", 5000
CURRENCY = "INR"
LOCALE = "en-IN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("expensex-ai-india")

app = Flask(__name__)
CORS(app)

def format_inr(value):
    return format_currency(value, CURRENCY, locale=LOCALE)

def normalize_ticker(ticker: str) -> str:
    """Convert simple ticker to NSE format by appending .NS"""
    ticker = ticker.upper().strip()
    if not ticker.endswith('.NS'):
        ticker = f"{ticker}.NS"
    return ticker

def load_models() -> Dict[str, object]:
    """Load models and create mapping from simple ticker to model"""
    models = {}
    if not MODELS_DIR.exists():
        log.error("models/ directory missing")
        return models
    for h5_path in MODELS_DIR.glob("*.h5"):
        full_ticker = h5_path.stem  # e.g., "TCS.NS"
        simple_ticker = full_ticker.replace('.NS', '')  # e.g., "TCS"
        try:
            models[simple_ticker] = load_model(h5_path)
            log.info("Loaded model for %s (file: %s)", simple_ticker, full_ticker)
        except Exception as exc:
            log.warning("Skipping %s: %s", full_ticker, exc)
    return models

def fetch_recent(ticker: str) -> Optional[Tuple[pd.Series, Dict]]:
    """Fetch recent data, automatically adding .NS suffix"""
    try:
        full_ticker = normalize_ticker(ticker)
        tk = yf.Ticker(full_ticker)
        hist = tk.history(period="90d")["Close"].dropna()
        info = tk.info
        return (hist, info) if len(hist) >= SEQ_LEN else None
    except Exception as exc:
        log.debug("fetch %s failed: %s", ticker, exc)
        return None

def predict_price(ticker: str, models: Dict) -> Optional[Dict]:
    """Predict price using simple ticker (without .NS)"""
    simple_ticker = ticker.upper().strip()
    if simple_ticker not in models:
        return None
    data = fetch_recent(simple_ticker)  # This will auto-add .NS
    if not data:
        return None
    prices, info = data
    last_price = float(prices.iloc[-1])
    scaler = MinMaxScaler()
    scaler.fit(prices.iloc[:-1].values.reshape(-1, 1))
    seq = scaler.transform(prices.values.reshape(-1, 1))[-SEQ_LEN:]
    input_data = seq.reshape(1, SEQ_LEN, 1)
    predicted_return = models[simple_ticker].predict(input_data, verbose=0)[0, 0]
    predicted_price = last_price * (1 + predicted_return)
    ret = predicted_return * 100
    return {
        "ticker": simple_ticker,  # Return simple ticker to frontend
        "current_price": last_price,
        "predicted_price": predicted_price,
        "predicted_return": ret,
        "sector": info.get("sector", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "historical_data": [
            {"date": str(d.date()), "price": float(p)} for d, p in prices.items()
        ],
    }

MODELS = load_models()
ALL_PREDICTIONS = []

def refresh_predictions():
    global ALL_PREDICTIONS
    ALL_PREDICTIONS = [p for p in (predict_price(t, MODELS) for t in MODELS) if p]
    log.info("Generated predictions for %d tickers", len(ALL_PREDICTIONS))

refresh_predictions()

@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "tickers": list(MODELS.keys()),
        "predictions_available": len(ALL_PREDICTIONS)
    })

@app.route("/analyze_portfolio", methods=["POST"])
def analyze_portfolio():
    try:
        data = request.get_json(silent=True) or {}
        cash = float(data.get("cash", 0))
        wallet = data.get("wallet", [])
        if cash < 0:
            return jsonify({"error": "Cash amount cannot be negative"}), 400
        holdings = []
        total_value = cash
        stock_returns = {}
        for h in wallet:
            ticker = h["ticker"].upper().strip()  # Simple ticker from frontend
            shares = float(h["shares"])
            purchase_price = float(h["purchase_price"])
            pred = predict_price(ticker, MODELS)
            if not pred:
                log.warning("No prediction available for %s", ticker)
                continue
            current_price = pred["current_price"]
            current_value = shares * current_price
            unrealized_gain = (current_price - purchase_price) * shares
            total_value += current_value
            holdings.append({
                "ticker": ticker,  # Keep simple ticker format
                "shares": shares,
                "purchase_price": purchase_price,
                "current_price": current_price,
                "current_value": current_value,
                "unrealized_gain": unrealized_gain,
                "allocation": 0
            })
            stock_returns[ticker] = pred["predicted_return"]
        
        # Rest of the function remains the same...
        for h in holdings:
            h["allocation"] = (h["current_value"] / total_value * 100) if total_value > 0 else 0
        for p in ALL_PREDICTIONS:
            if p["ticker"] not in stock_returns:
                stock_returns[p["ticker"]] = p["predicted_return"]
        
        # Continue with existing allocation logic...
        sorted_pos = sorted([(s, r) for s, r in stock_returns.items() if r > 0],
                            key=lambda x: x[1], reverse=True)
        tiers = [40, 30, 20, 10, 5, 5][:len(sorted_pos)]
        allocations = {}
        for i, (stock, _) in enumerate(sorted_pos):
            allocations[stock] = tiers[i] if i < len(tiers) else 0
        total_allocated = sum(allocations.values())
        if total_allocated < 100 and sorted_pos:
            allocations[sorted_pos[0][0]] += 100 - total_allocated
        
        allocation_details = {}
        total_projected_profit = 0
        if cash > 0:
            for ticker, percentage in allocations.items():
                if percentage > 0:
                    for p in ALL_PREDICTIONS:
                        if p["ticker"] == ticker:
                            current_price = p["current_price"]
                            projected_price = p["predicted_price"]
                            investment_amount = (cash * percentage) / 100
                            shares_to_buy = investment_amount / current_price
                            projected_profit = shares_to_buy * (projected_price - current_price)
                            total_projected_profit += projected_profit
                            allocation_details[ticker] = {
                                "percentage": percentage,
                                "investment_amount": investment_amount,
                                "shares_to_buy": round(shares_to_buy, 4),
                                "current_price": current_price,
                                "projected_price": projected_price,
                                "projected_profit": projected_profit
                            }
                            break
        
        wallet_tickers = {h["ticker"] for h in holdings}
        buy_candidates = [
            p for p in ALL_PREDICTIONS
            if p["ticker"] not in wallet_tickers and p["predicted_return"] > 2
        ]
        buy_candidates.sort(key=lambda x: x["predicted_return"], reverse=True)
        hold_candidates = [
            p for p in ALL_PREDICTIONS
            if p["ticker"] in wallet_tickers and -2 <= p["predicted_return"] <= 2
        ]
        sell_candidates = [
            p for p in ALL_PREDICTIONS
            if p["ticker"] in wallet_tickers and p["predicted_return"] < -2
        ]
        recommendations = {
            "buy": buy_candidates[:5],
            "hold": hold_candidates,
            "sell": sell_candidates
        }
        top_stocks = sorted(ALL_PREDICTIONS, key=lambda x: x["predicted_return"], reverse=True)[:10]
        response = {
            "success": True,
            "portfolio_summary": {
                "total_value": total_value,
                "available_cash": cash,
                "holdings": holdings
            },
            "recommendations": recommendations,
            "allocations": allocations,
            "allocation_details": allocation_details,
            "total_projected_profit": total_projected_profit,
            "top_stocks": top_stocks,
            "timestamp": datetime.utcnow().isoformat()
        }
        return jsonify(response)
    except ValueError as e:
        log.error("Value error in analyze_portfolio: %s", str(e))
        return jsonify({"error": f"Invalid input data: {str(e)}"}), 400
    except Exception as e:
        log.error("Error in analyze_portfolio: %s", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/stock_details/<ticker>")
def stock_details(ticker: str):
    ticker = ticker.upper().strip() 
    if not ticker:
        return jsonify({"error": "Missing ticker"}), 400
    try:
        data = predict_price(ticker, MODELS)
        if data:
            return jsonify(data), 200
        else:
            return jsonify({"error": f"No model or data available for {ticker}"}), 404
    except Exception as e:
        log.error("Error getting stock details for %s: %s", ticker, str(e))
        return jsonify({"error": "Failed to analyze stock"}), 500

@app.route("/refresh_predictions", methods=["POST"])
def refresh_predictions_endpoint():
    try:
        refresh_predictions()
        return jsonify({
            "success": True,
            "message": f"Refreshed predictions for {len(ALL_PREDICTIONS)} tickers"
        })
    except Exception as e:
        log.error("Error refreshing predictions: %s", str(e))
        return jsonify({"error": "Failed to refresh predictions"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    log.info("ExpenseXchange AI India Backend starting...")
    log.info("Available models: %s", list(MODELS.keys()))
    log.info("Serving on http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)