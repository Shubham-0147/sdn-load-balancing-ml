#!/usr/bin/env python3
"""
data_preprocessing.py: Load network traffic data, select features,
apply normalization and PCA for model training.
"""
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_data(file_path):
    """
    Load dataset from CSV file into a pandas DataFrame.
    The CSV is expected to contain traffic metrics (e.g., bandwidth usage, delay, loss).
    """
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape {df.shape} from {file_path}")
    return df

def select_features(df, target_column, k=5):
    """
    Select top k features based on univariate regression test (F-test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(score_func=f_regression, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_features}")
    return pd.DataFrame(X_new, columns=[X.columns[i] for i in selected_features])

def normalize_data(df, method="standard"):
    """
    Normalize the dataframe values either with StandardScaler or MinMaxScaler.
    """
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, columns=df.columns)
    print(f"Data normalized using {method} scaler.")
    return df_scaled

def apply_pca(df, n_components=3):
    """
    Reduce dimensionality of features using PCA.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df)
    cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(data=principal_components, columns=cols)
    print(f"PCA applied: reduced to {n_components} components.")
    return df_pca

def main():
    parser = argparse.ArgumentParser(description="Preprocess network traffic data.")
    parser.add_argument("--input", required=True, help="Path to input CSV data file.")
    parser.add_argument("--output", required=True, help="Path to save processed CSV.")
    args = parser.parse_args()

    config = load_config()
    test_size = config['preprocessing'].get('test_size', 0.2)

    # Load raw data
    df = load_data(args.input)

    # Example: assume 'target' column exists for supervision (could be next-time traffic)
    if 'target' in df.columns:
        target_col = 'target'
    else:
        # if no explicit target, create dummy or last column as target
        target_col = df.columns[-1]

    # Feature selection (optional)
    df_features = select_features(df, target_col, k=config['preprocessing'].get('pca_components', 5))
    df_features[target_col] = df[target_col]

    # Normalize
    df_norm = normalize_data(df_features, method="standard")

    # Dimensionality reduction
    df_pca = apply_pca(df_norm.drop(columns=[target_col]), n_components=config['preprocessing'].get('pca_components', 5))
    df_pca[target_col] = df_norm[target_col].values

    # Save processed data
    df_pca.to_csv(args.output, index=False)
    print(f"Processed data saved to {args.output}")

if __name__ == "__main__":
    main()
