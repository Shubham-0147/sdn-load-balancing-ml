#!/usr/bin/env python3
"""
train_rf.py: Train a Random Forest model for path selection in SDN.
"""
import pandas as pd
import argparse
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_random_forest(data_path, model_path, target_column='best_path', n_estimators=100):
    # Load data
    df = pd.read_csv(data_path)
    print(f"Training data shape: {df.shape}")

    # Features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    # Define and train Random Forest
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest accuracy on test set: {acc:.4f}")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    print(f"Random Forest model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest for SDN path selection.")
    parser.add_argument("--data", required=True, help="Path to CSV data with features and target 'best_path'.")
    parser.add_argument("--model_path", required=True, help="Path to save trained RF model (.pkl).")
    args = parser.parse_args()

    train_random_forest(args.data, args.model_path)
