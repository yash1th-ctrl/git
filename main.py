import torch
import numpy as np
import random
from configs.config import Config
from data.preprocessing import load_data
from models.cnn import SimpleCNN, CNN_LSTM
from fl_core.client import Client
from fl_core.server import Server
import copy
import os
import time

def main():
    print("Starting Lightweight FL IDS Simulation...")
    Config.log_config()
    
    # 1. Load Data
    print("Loading Data...")
    client_loaders, test_loader, input_dim = load_data(
        Config.DATA_PATH, 
        Config.NUM_CLIENTS, 
        Config.NON_IID
    )
    print(f"Data Loaded. Input Dimension: {input_dim}")
    
    # 2. Initialize Model
    if Config.MODEL_TYPE == "CNN":
        global_model = SimpleCNN(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    elif Config.MODEL_TYPE == "CNN_LSTM":
        global_model = CNN_LSTM(input_dim=input_dim, num_classes=Config.NUM_CLASSES)
    else:
        raise ValueError("Unknown Model Type")
        
    # 3. Initialize Server
    server = Server(global_model, test_loader, Config.DEVICE, Config)
    
    # 4. Initialize Clients
    clients = []
    for i in range(Config.NUM_CLIENTS):
        # Each client gets a copy of the model structure (weights will be overwritten)
        client_model = copy.deepcopy(global_model)
        client = Client(i, client_model, client_loaders[i], Config.DEVICE, Config)
        clients.append(client)
        
        # Debug: Check label distribution
        labels = []
        for _, y in client_loaders[i]:
            labels.extend(y.numpy())
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Client {i} Label Distribution: {dict(zip(unique, counts))}")
        
    # Save a temp model to measure size
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    torch.save(global_model.state_dict(), f"{Config.OUTPUT_DIR}/temp_model.pth")
    
    total_comm_overhead = 0.0
        
    # 5. FL Training Loop
    print("\n--- Starting Training Rounds ---")
    for round_idx in range(Config.ROUNDS):
        print(f"\nRound {round_idx + 1}/{Config.ROUNDS}")
        
        # Client Selection (Random for now)
        selected_clients = random.sample(clients, Config.CLIENTS_PER_ROUND)
        
        client_weights = []
        client_losses = []
        client_accs = []
        
        # Local Training
        for client in selected_clients:
            # Sync with global model
            client.model.load_state_dict(server.global_model.state_dict())
            
            # Train
            payload, loss, acc = client.train()
            
            # --- Security: Authentication Check ---
            if payload["auth_token"] != Config.AUTH_TOKEN:
                print(f"  [SECURITY ALERT] Client {client.client_id} Authentication Failed! Dropping.")
                continue
                
            # --- Security: Poisoning Check (Norm Threshold) ---
            # Calculate norm of weight difference
            w = payload["weights"]
            is_poisoned = False
            # Simple check: if any weight tensor has extreme values
            for k in w.keys():
                if torch.isnan(w[k]).any() or torch.max(torch.abs(w[k])) > 10.0: # Arbitrary large bound
                     is_poisoned = True
                     break
            
            if is_poisoned:
                print(f"  [SECURITY ALERT] Client {client.client_id} Update Rejected (Poisoning Detected).")
                continue
            
            client_weights.append(w)
            client_losses.append(loss)
            client_accs.append(acc)
            
            print(f"  Client {client.client_id}: Loss={loss:.4f}, Acc={acc:.2f}%")
            
        # Aggregation
        if len(client_weights) > 0:
            start_time = time.time()
            server.aggregate(client_weights)
            agg_time = time.time() - start_time
        else:
            print("  No valid updates this round.")
            agg_time = 0
        
        # Calculate Communication Overhead
        # Model size in MB * Number of clients * 2 (Upload + Download)
        # 2 because clients download global model and upload local model
        model_size_mb = os.path.getsize(f"{Config.OUTPUT_DIR}/temp_model.pth") / (1024 * 1024) if os.path.exists(f"{Config.OUTPUT_DIR}/temp_model.pth") else 0.1 # Approx
        round_comm_overhead = model_size_mb * len(selected_clients) * 2
        total_comm_overhead += round_comm_overhead
        
        # Global Evaluation
        start_eval = time.time()
        val_acc, val_loss, val_f1, val_prec, val_rec = server.evaluate()
        eval_time = time.time() - start_eval
        
        print(f"  Global Model: Loss={val_loss:.4f}, Acc={val_acc:.2f}%, F1={val_f1:.4f}")
        print(f"  Comm. Overhead: {total_comm_overhead:.2f} MB (Total)")
        print(f"  Latency: Aggregation={agg_time*1000:.2f}ms, Inference={eval_time*1000:.2f}ms")
        
    print("\nSimulation Complete.")
    
    # Save Model
    if not os.path.exists(Config.OUTPUT_DIR):
        os.makedirs(Config.OUTPUT_DIR)
    torch.save(server.global_model.state_dict(), f"{Config.OUTPUT_DIR}/global_model.pth")
    print(f"Model saved to {Config.OUTPUT_DIR}/global_model.pth")

if __name__ == "__main__":
    main()
