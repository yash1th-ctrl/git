# Implementation Plan: Lightweight Federated Learning-based IDS

This plan outlines the development steps for the Lightweight Federated Learning-based Intrusion Detection System (IDS) for IoT devices, covering Phase I and Phase II objectives.

## Phase I: Foundation & Baseline (S7)

### 1. Project Setup & Architecture
- [ ] Initialize project structure (directories for data, models, fl_core, utils).
- [ ] Define configuration management (hyperparameters, paths).
- [ ] Set up Python environment and dependencies (`requirements.txt`).

### 2. Data Preprocessing & Partitioning
- [ ] Implement data loaders for TON-IoT, UNSW-NB15, CIC-IDS-2017.
- [ ] Implement preprocessing pipeline (normalization, encoding).
- [ ] **Novelty**: Implement Non-IID data partitioning strategy to simulate heterogeneous IoT clients.

### 3. Model Development
- [ ] Implement baseline CNN architecture suitable for IoT traffic data.
- [ ] Implement CNN+LSTM architecture (for later comparison).
- [ ] Create local training loop (Client side).

### 4. Federated Learning Engine (Baseline)
- [ ] Implement `Client` class: Local training, weight updates.
- [ ] Implement `Server` class: Global aggregation (FedAvg).
- [ ] Implement the main simulation loop (Rounds, Client selection).

### 5. Preliminary Evaluation
- [ ] Implement metrics: Accuracy, Precision, Recall, F1-Score.
- [ ] Implement communication overhead tracking (model size * updates).
- [ ] Create basic visualization of training progress.

## Phase II: Optimization, Privacy & Robustness (S8)

### 6. Lightweight Optimization (Novelty)
- [ ] **Pruning**: Implement structured/unstructured pruning for the CNN model.
- [ ] **Quantization**: Implement INT8 quantization (Post-training or Quantization-aware training).
- [ ] Measure latency and model size reduction.

### 7. Advanced Aggregation & Robustness (Novelty)
- [ ] Implement **FedProx** for handling data heterogeneity.
- [ ] Implement **Trimmed Mean** / **Krum** for defense against poisoning.
- [ ] Simulate label-flipping attacks to test robustness.

### 8. Privacy & Security (Novelty)
- [ ] Implement **Differential Privacy (DP-SGD)** (clipping gradients + noise injection).
- [ ] Simulate basic token-based authentication for clients.

### 9. Adaptive Client Selection (Novelty)
- [ ] Implement logic to select clients based on "resource availability" (simulated) or loss thresholds.

### 10. Deployment & Final Benchmarking
- [ ] Create a "Deployment" script to run inference on a single node (simulating Edge device).
- [ ] **Energy Profiling**: Integrate tools to estimate energy consumption (e.g., using `codecarbon` or approximate FLOPs-based calculation).
- [ ] Generate final comparison reports and dashboards.

## Directory Structure
```
Lightweight_FL_IDS/
├── data/               # Dataset handlers and splitters
├── models/             # CNN, LSTM model definitions
├── fl_core/            # Server, Client, Aggregation logic
├── utils/              # Metrics, Plotting, Quantization helpers
├── configs/            # Config files
├── experiments/        # Logs and results
├── main.py             # Entry point for simulation
└── requirements.txt    # Dependencies
```
