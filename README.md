# Dynamic Load Balancing in SDN Using Machine Learning

This project implements a **dynamic load balancing** strategy in a Software-Defined Networking (SDN) environment using machine learning. The solution includes:

- **Traffic Forecasting (LSTM)**: Predict future traffic load based on historical patterns.  
- **Path Selection (Random Forest)**: Choose optimal routing paths using a trained Random Forest classifier/regressor.  
- **SDN Controller (Ryu)**: Monitors network traffic in real time and dynamically reroutes flows via OpenFlow rules.  
- **Network Emulation (Mininet)**: Emulate network topologies and run the Ryu controller.  
- **Visualization**: Dashboard/plots for throughput, latency, packet loss, and other metrics.  
- **Evaluation**: Compute performance metrics (throughput, latency, packet loss, jitter, prediction accuracy).


