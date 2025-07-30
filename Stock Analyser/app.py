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

MODELS_DIR = Path("models")
SEQ_LEN = 60
HOST, PORT = "0.0.0.0", 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("expensex-ai")

app = Flask(__name__)
CORS(app)


def load_models() -> Dict[str, object]:
    """Load all available ML models from the models directory."""
    models = {}
    if not MODELS_DIR.exists():
        log.error("models/ directory missing")
        return models

    for h5_path in MODELS_DIR.glob("*.h5"):
        ticker = h5_path.stem
        try:
            models[ticker] = load_model(h5_path)
            log.info("Loaded model for %s", ticker)
        except Exception as exc:
            log.warning("Skipping %s: %s", ticker, exc)
    return models


def fetch_recent(ticker: str) -> Optional[Tuple[pd.Series, Dict]]:
    """Fetch recent stock data and info for a given ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="90d")["Close"].dropna()
        info = tk.info
        return (hist, info) if len(hist) >= SEQ_LEN else None
    except Exception as exc:
        log.debug("fetch %s failed: %s", ticker, exc)
        return None


def predict_price(ticker: str, models: Dict) -> Optional[Dict]:
    """Predict future price for a given ticker using the loaded model."""
    if ticker not in models:
        return None

    data = fetch_recent(ticker)
    if not data:
        return None

    prices, info = data
    last_price = float(prices.iloc[-1])
    
    # Scale only using past data — EXCLUDING the current day
    scaler = MinMaxScaler()
    scaler.fit(prices.iloc[:-1].values.reshape(-1, 1))

    # Get the last SEQ_LEN days before the last known price
    seq = scaler.transform(prices.values.reshape(-1, 1))[-SEQ_LEN:]
    input_data = seq.reshape(1, SEQ_LEN, 1)

    predicted_return = models[ticker].predict(input_data, verbose=0)[0, 0]
    predicted_price = last_price * (1 + predicted_return)
    ret = predicted_return * 100

    return {
        "ticker": ticker,
        "current_price": last_price,
        "predicted_price": predicted_price,
        "predicted_return": ret,
        "sector": info.get("sector", "Unknown"),
        "market_cap": info.get("marketCap", 0),
        "historical_data": [
            {"date": str(d.date()), "price": float(p)} for d, p in prices.items()
        ],
    }


def calculate_allocations(stock_returns: dict) -> dict:
    """Calculate suggested portfolio allocations based on predicted returns.
    
    This function implements the exact same logic as gru_test.py:
    - Sort stocks by return percentage (descending)
    - Separate positive and negative returns
    - Allocation tiers only for positive returns: [40%, 30%, 20%, 10%]
    - Allocate 0% to negative return stocks
    - Redistribute any remaining percentage to the top performer
    """
    # Sort stocks by return percentage (descending)
    sorted_stocks = sorted(stock_returns.items(), key=lambda x: x[1], reverse=True)
    
    # Separate positive and negative returns
    positive = [(s, r) for s, r in sorted_stocks if r >= 0]
    negative = [(s, r) for s, r in sorted_stocks if r < 0]
    
    # Allocation tiers only for positive returns
    tiers = [40, 30, 20, 10][:len(positive)]
    allocations = {}
    
    # Allocate to positive return stocks
    for i, (stock, return_pct) in enumerate(positive):
        allocations[stock] = tiers[i] if i < len(tiers) else 0
    
    # Allocate 0% to negative return stocks
    for stock, _ in negative:
        allocations[stock] = 0
    
    # Redistribute any remaining percentage
    total_allocated = sum(allocations.values())
    if total_allocated < 100 and positive:
        allocations[positive[0][0]] += 100 - total_allocated
    
    return allocations


# Load models and generate predictions at startup
MODELS = load_models()
ALL_PREDICTIONS = []

def refresh_predictions():
    """Refresh all predictions for available tickers."""
    global ALL_PREDICTIONS
    ALL_PREDICTIONS = [p for p in (predict_price(t, MODELS) for t in MODELS) if p]
    log.info("Generated predictions for %d tickers", len(ALL_PREDICTIONS))

# Initial prediction generation
refresh_predictions()


@app.route("/status")
def status():
    """Health check endpoint."""
    return jsonify({
        "status": "ok", 
        "tickers": list(MODELS.keys()),
        "predictions_available": len(ALL_PREDICTIONS)
    })


@app.route("/analyze_portfolio", methods=["POST"])
def analyze_portfolio():
    """Analyze portfolio and provide recommendations."""
    try:
        data = request.get_json(silent=True) or {}
        cash = float(data.get("cash", 0))
        wallet = data.get("wallet", [])

        if cash < 0:
            return jsonify({"error": "Cash amount cannot be negative"}), 400

        holdings = []
        total_value = cash
        stock_returns = {}

        # Process existing holdings
        for h in wallet:
            ticker = h["ticker"].upper()
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
                "ticker": ticker,
                "shares": shares,
                "purchase_price": purchase_price,
                "current_price": current_price,
                "current_value": current_value,
                "unrealized_gain": unrealized_gain,
                "allocation": 0  # Will be calculated below
            })
            
            stock_returns[ticker] = pred["predicted_return"]

        # Calculate allocation percentages
        for h in holdings:
            h["allocation"] = (h["current_value"] / total_value * 100) if total_value > 0 else 0

        # Add all available predictions to stock_returns for allocation calculation
        for p in ALL_PREDICTIONS:
            if p["ticker"] not in stock_returns:
                stock_returns[p["ticker"]] = p["predicted_return"]

        # Generate allocations
        allocations = calculate_allocations(stock_returns)
        
        # Calculate detailed allocation breakdown with amounts and shares
        # ONLY use available cash for new investments, not total portfolio value
        allocation_details = {}
        if cash > 0:  # Only allocate if there's available cash
            for ticker, percentage in allocations.items():
                if percentage > 0:
                    # Find the current price for this ticker
                    current_price = None
                    for p in ALL_PREDICTIONS:
                        if p["ticker"] == ticker:
                            current_price = p["current_price"]
                            break
                    
                    if current_price:
                        # Use only available cash for allocation, not total portfolio value
                        investment_amount = (cash * percentage) / 100
                        shares_to_buy = investment_amount / current_price
                        
                        allocation_details[ticker] = {
                            "percentage": percentage,
                            "investment_amount": investment_amount,
                            "shares_to_buy": shares_to_buy,
                            "current_price": current_price
                        }

        # Generate recommendations
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

        # Get top performing stocks
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
    """Get detailed analysis for a specific stock ticker."""
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
    """Manual endpoint to refresh all predictions."""
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
    log.info("ExpenseXchange AI Backend starting...")
    log.info("Available models: %s", list(MODELS.keys()))
    log.info("Serving on http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, debug=True, use_reloader=False)