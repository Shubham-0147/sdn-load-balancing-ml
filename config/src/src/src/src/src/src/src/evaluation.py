#!/usr/bin/env python3
"""
evaluation.py: Calculate and report network performance metrics.
"""
import pandas as pd
import numpy as np
import argparse
from src.utils import compute_throughput, compute_latency, compute_packet_loss, compute_jitter

def evaluate_performance(log_file):
    """
    Reads a CSV log with columns: time, sent_bytes, received_bytes, latency_ms (list or avg), loss_packets.
    Computes throughput, avg latency, packet loss %, jitter.
    """
    df = pd.read_csv(log_file)
    # Example calculations (assuming log has these columns)
    total_time = df['time'].iloc[-1] - df['time'].iloc[0]
    total_bytes = df['received_bytes'].sum()
    thr = compute_throughput(total_bytes, total_time)
    print(f"Overall throughput: {thr/1e6:.3f} Mbps")

    # Latency (assuming 'latency_ms' is average latency per interval)
    avg_latency = compute_latency(df['latency_ms'])
    print(f"Average latency: {avg_latency:.2f} ms")

    # Jitter (std deviation of latency)
    jitter = compute_jitter(df['latency_ms'])
    print(f"Latency jitter (std dev): {jitter:.2f} ms")

    # Packet loss (assuming 'sent_packets' and 'recv_packets')
    if 'sent_packets' in df.columns and 'recv_packets' in df.columns:
        sent = df['sent_packets'].sum()
        recv = df['recv_packets'].sum()
        loss = compute_packet_loss(sent, recv)
        print(f"Packet loss: {loss:.2f} %")
    else:
        print("Sent/received packet counts not found; cannot compute packet loss.")

    # Predictive accuracy (if actual vs predicted available)
    if 'predicted' in df.columns and 'actual' in df.columns:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(df['actual'], df['predicted'])
        mae = mean_absolute_error(df['actual'], df['predicted'])
        print(f"Prediction MSE: {mse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate network performance metrics.")
    parser.add_argument("--log_file", required=True, help="CSV file with performance logs.")
    args = parser.parse_args()
    evaluate_performance(args.log_file)
