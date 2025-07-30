import os
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler
import optuna
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import csv, os

logging.basicConfig(level=logging.INFO)

RESULTS_DIR = Path("results")
CHARTS_DIR  = RESULTS_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
EPOCHS = 100
MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(exist_ok=True)
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA']

METRICS_CSV = RESULTS_DIR / "metrics.csv"
# create CSV with header if it does not exist
if not METRICS_CSV.exists():
    with METRICS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "MAE", "RMSE", "R2"])

def download_data(ticker):
    df = yf.download(ticker, period="5y")
    return df["Close"] if "Close" in df else None

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len + 1, len(data)):
        prev = data[i - 1, 0]
        curr = data[i, 0]

        if prev <= 0 or curr <= 0 or np.isnan(prev) or np.isnan(curr):
            continue

        X.append(data[i - seq_len - 1:i - 1, 0])
        y.append(np.log(curr / prev))

    return np.array(X), np.array(y)


def build_model(trial, input_shape):
    model = Sequential()
    
    units = trial.suggest_int("units", 32, 128, step=32)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "nadam"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    model.add(GRU(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    if optimizer_name == "adam":
        from tensorflow.keras.optimizers import Adam
        optimizer = Adam(learning_rate=lr)
    elif optimizer_name == "rmsprop":
        from tensorflow.keras.optimizers import RMSprop
        optimizer = RMSprop(learning_rate=lr)
    else:
        from tensorflow.keras.optimizers import Nadam
        optimizer = Nadam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=Huber())
    return model

def objective(trial, series):
    seq_len = trial.suggest_int("sequence_length", 60, 180, step=30)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_model(trial, input_shape=(seq_len, 1))
    es = EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        callbacks=[es],
        verbose=0
    )

    return min(history.history["val_loss"])

def train_with_optuna(ticker):
    logging.info(f"Optimizing {ticker}")
    series = download_data(ticker)
    if series is None or len(series) < 200:
        logging.warning(f"Skipping {ticker}: insufficient data.")
        return

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, series), n_trials=60, n_jobs = 4)

    logging.info(f"Best trial for {ticker}: {study.best_trial.params}")

    best_trial = study.best_trial
    seq_len = best_trial.params["sequence_length"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = create_sequences(data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = build_model(best_trial, input_shape=(seq_len, 1))
    es = EarlyStopping(patience=10, restore_best_weights=True)
    
    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=best_trial.params["batch_size"],
        callbacks=[es],
        verbose=0
    )

    model.save(MODELS_DIR / f"{ticker}.h5")
    logging.info(f"Saved model for {ticker}")

def main():
    for ticker in TICKERS:
        try:
            train_with_optuna(ticker)
        except Exception as e:
            logging.error(f"Failed on {ticker}: {e}")
            
def evaluate_and_save(ticker, series, model, best_params):
    seq_len = best_params["sequence_length"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(data, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]
    y_pred = model.predict(X_test, verbose=0).flatten()

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # 1. append row to CSV
    with METRICS_CSV.open("a", newline="") as f:
        csv.writer(f).writerow([ticker, mae, rmse, r2])

    # 2. save chart
    plt.figure(figsize=(6, 3))
    plt.title(f"{ticker} – actual vs predicted log-return (test)")
    plt.plot(y_test, label="actual")
    plt.plot(y_pred, label="predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / f"{ticker}_perf.png", dpi=150)
    plt.close()

    logging.info("%s | MAE: %.4f | RMSE: %.4f | R2: %.4f | saved", ticker, mae, rmse, r2)

def evaluate_all():
    for ticker in TICKERS:
        try:
            series = download_data(ticker)
            if series is None or len(series) < 200:
                continue
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda t: objective(t, series), n_trials=1, n_jobs=1)
            model_path = MODELS_DIR / f"{ticker}.h5"
            if not model_path.exists():
                continue
            model = load_model(model_path)
            evaluate_and_save(ticker, series, model, study.best_params)
        except Exception as e:
            logging.warning("Evaluation failed for %s: %s", ticker, e)

if __name__ == "__main__":
    main()   
    evaluate_all() 
