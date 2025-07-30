# Stock Analyser

This project is a web-based stock analysis tool that uses machine learning to predict future stock prices. It provides a simple interface to visualize historical stock data and view predictions.

## Features

*   **Historical Data Visualization:** View historical stock data (Open, High, Low, Close) in an interactive chart.
*   **Stock Price Prediction:** Get stock price predictions based on a trained GRU model.
*   **Supports Multiple Stocks:** Pre-trained models are available for several popular stocks (AAPL, AMZN, GOOG, META, MSFT, NVDA, TSLA).

## Technology Stack

*   **Backend:**
    *   Python
    *   Flask
    *   TensorFlow/Keras
    *   yfinance
    *   NumPy
    *   Pandas
*   **Frontend:**
    *   HTML
    *   JavaScript
    *   Chart.js

## Getting Started

### Prerequisites

*   Python 3.x
*   pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/stock-analyser.git
    cd stock-analyser
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r "Stock Analyser/requirements.txt"
    ```

### Running the Application

1.  **Start the Flask server:**
    ```bash
    python "Stock Analyser/app.py"
    ```

2.  **Open your browser:**
    Navigate to `http://127.0.0.1:5000`

## Usage

1.  Select a stock from the dropdown menu.
2.  The historical data for the selected stock will be displayed in a chart.
3.  The predicted next-day closing price will be shown below the chart.

## Model Training

The GRU models were trained using the `train_model.py` script. This script fetches historical stock data using the `yfinance` library, preprocesses the data, and trains an GRU model for each stock. The trained models are saved in the `Models/` directory.

To train a new model, you can modify the `train_model.py` script and run it:

```bash
python "Stock Analyser/train_model.py"
```

## Project Structure

```
.
├── Models/
│   ├── AAPL.h5
│   ├── ... (other trained models)
├── Stock Analyser/
│   ├── aixt3.html      # Frontend HTML
│   ├── app.js          # Frontend JavaScript
│   ├── app.py          # Flask application
│   ├── README.md       # This file
│   ├── requirements.txt# Python dependencies
│   └── train_model.py  # Model training script
```