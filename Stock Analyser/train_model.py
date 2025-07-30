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


logging.basicConfig(level=logging.INFO)

RESULTS_DIR = Path("results")
CHARTS_DIR  = RESULTS_DIR / "charts"
RESULTS_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
EPOCHS = 100
MODELS_DIR = Path("Models")
MODELS_DIR.mkdir(exist_ok=True)
TICKERS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'META', 'TSLA']


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
            

if __name__ == "__main__":
    main()   
