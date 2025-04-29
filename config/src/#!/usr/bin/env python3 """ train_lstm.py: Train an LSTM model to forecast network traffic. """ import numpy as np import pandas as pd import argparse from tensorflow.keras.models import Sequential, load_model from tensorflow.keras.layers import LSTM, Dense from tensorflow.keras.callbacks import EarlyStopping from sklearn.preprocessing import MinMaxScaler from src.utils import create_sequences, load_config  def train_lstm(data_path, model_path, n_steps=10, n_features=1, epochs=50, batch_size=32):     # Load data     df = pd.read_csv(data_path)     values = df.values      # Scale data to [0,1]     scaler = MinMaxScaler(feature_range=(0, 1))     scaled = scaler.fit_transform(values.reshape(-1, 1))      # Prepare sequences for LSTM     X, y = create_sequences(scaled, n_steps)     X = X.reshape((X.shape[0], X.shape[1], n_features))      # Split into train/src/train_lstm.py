#!/usr/bin/env python3
"""
train_lstm.py: Train an LSTM model to forecast network traffic.
"""
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from src.utils import create_sequences, load_config

def train_lstm(data_path, model_path, n_steps=10, n_features=1, epochs=50, batch_size=32):
    # Load data
    df = pd.read_csv(data_path)
    values = df.values

    # Scale data to [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values.reshape(-1, 1))

    # Prepare sequences for LSTM
    X, y = create_sequences(scaled, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # Split into train/test
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    # Train model
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, 
              validation_data=(X_test, y_test),
              epochs=epochs, batch_size=batch_size,
              callbacks=[es], verbose=2)

    # Evaluate
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Test Loss: {loss:.4f}")

    # Save model
    model.save(model_path)
    print(f"LSTM model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM model for traffic forecasting.")
    parser.add_argument("--data", required=True, help="Path to preprocessed data CSV.")
    parser.add_argument("--model_path", required=True, help="File path to save the trained LSTM model (HDF5).")
    parser.add_argument("--n_steps", type=int, default=10, help="Number of time steps for sequences.")
    parser.add_argument("--n_features", type=int, default=1, help="Number of features for LSTM input.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    train_lstm(args.data, args.model_path, n_steps=args.n_steps,
               n_features=args.n_features, epochs=args.epochs, batch_size=args.batch_size)
