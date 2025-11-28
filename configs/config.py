import torch

class Config:
    # Project Details
    PROJECT_NAME = "Lightweight_FL_IDS"
    OUTPUT_DIR = "./experiments"
    
    # Data
    DATASET_NAME = "UNSW-NB15" # Options: "UNSW-NB15", "TON-IoT", "CIC-IDS-2017"
    DATA_PATH = "./data/UNSW_NB15_sample.csv" # Path to .csv file (e.g., ./data/CIC-IDS-2017.csv)
    NUM_CLIENTS = 2 # Reduced for sample run
    NON_IID = False
    ALPHA = 0.5 # Dirichlet distribution parameter for non-IID split
    
    # Model
    MODEL_TYPE = "CNN" # or "CNN_LSTM"
    INPUT_SHAPE = (1, 784) # Placeholder, needs to match feature size
    NUM_CLASSES = 2 # Binary classification (Attack vs Normal) or Multi-class
    
    # Federated Learning
    ROUNDS = 5 # Reduced for sample run
    CLIENTS_PER_ROUND = 2
    EPOCHS_PER_CLIENT = 3
    BATCH_SIZE = 32
    LR = 0.01
    
    # Optimization (Phase II)
    PRUNING_AMOUNT = 0.2
    QUANTIZATION = False
    
    # Privacy & Robustness (Phase II -> Moved to S7 Basic)
    DP_SIGMA = 0.01 # Differential Privacy noise multiplier (Small for S7 demo)
    DP_CLIP = 1.0   # Gradient clipping threshold
    AUTH_TOKEN = "secure_token_123" # Shared secret for authentication
    MAX_UPDATE_NORM = 2.0 # Threshold for poisoning check
    AGGREGATION = "FedAvg" # FedAvg, FedProx, Krum
    
    # Hardware Simulation
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def log_config():
        print(f"Project: {Config.PROJECT_NAME}")
        print(f"Device: {Config.DEVICE}")
        print(f"Model: {Config.MODEL_TYPE}, Aggregation: {Config.AGGREGATION}")
