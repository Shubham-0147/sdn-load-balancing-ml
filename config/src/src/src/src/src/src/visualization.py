#!/usr/bin/env python3
"""
visualization.py: Plot throughput, latency, packet loss, etc. from logs.
"""
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def plot_throughput(time, throughput_values):
    plt.figure()
    plt.plot(time, throughput_values, marker='o')
    plt.title("Throughput Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Throughput (Mbps)")
    plt.grid(True)
    plt.savefig("throughput.png")
    print("Throughput plot saved to throughput.png")
    plt.close()

def plot_latency(time, latency_values):
    plt.figure()
    plt.plot(time, latency_values, marker='x', color='orange')
    plt.title("Latency Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.savefig("latency.png")
    print("Latency plot saved to latency.png")
    plt.close()

def plot_packet_loss(time, loss_values):
    plt.figure()
    plt.plot(time, loss_values, marker='s', color='red')
    plt.title("Packet Loss Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Packet Loss (%)")
    plt.grid(True)
    plt.savefig("packet_loss.png")
    print("Packet loss plot saved to packet_loss.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate performance plots from log data.")
    parser.add_argument("--log_file", required=True, help="CSV file with time-series metrics: time, throughput, latency, packet_loss")
    args = parser.parse_args()

    df = pd.read_csv(args.log_file)
    time = df['time']
    plot_throughput(time, df['throughput'])
    plot_latency(time, df['latency'])
    plot_packet_loss(time, df['packet_loss'])

if __name__ == "__main__":
    main()
