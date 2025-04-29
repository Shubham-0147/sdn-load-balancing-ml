#!/usr/bin/env python3
"""
utils.py: Helper functions for SDN load balancing project.
"""
import numpy as np
import yaml
import networkx as nx

def load_config(config_path="config/config.yaml"):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)

def create_sequences(data, n_steps):
    """
    Convert a time series array into input-output sequences for LSTM.
    data: numpy array of shape (num_samples, 1) or (num_samples,)
    n_steps: number of timesteps in each input sequence.
    Returns: X (samples, n_steps), y (samples,)
    """
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    X = np.array(X)
    y = np.array(y)
    return X, y

def load_models(lstm_path, rf_path):
    """
    Load the trained LSTM and Random Forest models from disk.
    """
    from tensorflow.keras.models import load_model
    import pickle
    lstm_model = load_model(lstm_path)
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    return lstm_model, rf_model

def build_network_graph(adjacency_list):
    """
    Build a directed networkx graph from adjacency information.
    adjacency_list: list of (src, dst, port_src_to_dst) tuples.
    Returns a DiGraph with port information stored as edge attribute.
    """
    G = nx.DiGraph()
    for src, dst, port in adjacency_list:
        G.add_edge(src, dst, port=port)
    return G

def compute_throughput(bytes_transferred, time_seconds):
    """
    Compute throughput in bits per second.
    """
    return (bytes_transferred * 8) / time_seconds

def compute_latency(delay_list):
    """
    Compute average latency from a list of delays (in milliseconds).
    """
    return np.mean(delay_list)

def compute_packet_loss(sent_packets, received_packets):
    """
    Compute packet loss percentage.
    """
    if sent_packets == 0:
        return 0
    loss = (sent_packets - received_packets) / sent_packets * 100
    return loss

def compute_jitter(latency_values):
    """
    Compute jitter (variation in latency) as standard deviation.
    """
    return np.std(latency_values)
