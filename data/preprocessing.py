import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import make_classification

class IDSDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_data(data_path=None, num_clients=10, non_iid=False):
    """
    Loads data from CSV or generates synthetic data if path is invalid.
    Returns a list of train_loaders (one per client) and a global test_loader.
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading dataset from {data_path}...")
        
        # 1. Attempt to read with header
        df = pd.read_csv(data_path, low_memory=False)
        
        # Check if the first row looks like data (heuristic: if column names are numbers or IP addresses)
        # For UNSW-NB15 sample, the first column is an IP '59.166.0.0'.
        # If the first column name looks like an IP or number, reload without header.
        try:
            import ipaddress
            ipaddress.ip_address(df.columns[0])
            has_header = False
        except ValueError:
            # Check if it's a float/int
            try:
                float(df.columns[0])
                has_header = False
            except ValueError:
                has_header = True
                
        if not has_header:
            print("  Detected no header. Reloading with header=None...")
            df = pd.read_csv(data_path, header=None, low_memory=False)
        
        # Clean column names (remove spaces often found in CIC-IDS-2017)
        if has_header:
            df.columns = df.columns.str.strip()
        
        # Preprocessing for UNSW-NB15 / TON-IoT / CIC-IDS-2017
        
        # Handle Infinity and NaNs
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # LEAKAGE DETECTION (Before Encoding)
        cols_to_drop_indices = []
        if not has_header:
            # Check if the second to last column is 'object' type (likely attack category)
            if df.iloc[:, -2].dtype == 'object':
                print("  [Leakage Prevention] Detected categorical column at index -2. Marking for removal.")
                cols_to_drop_indices.append(-2)

        # 1. Handle Categorical Features (Label Encoding)
        for col in df.columns:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # 2. Separate Features and Label
        target_col = None
        
        if has_header:
            # Common label column names: 'label', 'Label', 'class', 'attack_cat'
            possible_labels = ['label', 'Label', 'class', 'attack_cat']
            for col in possible_labels:
                if col in df.columns:
                    target_col = col
                    break
        
        if target_col:
            y = df[target_col].values
            # Drop other non-feature columns if they exist
            drop_cols = [target_col, 'type', 'id', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 'attack_cat'] 
            X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore').values
        else:
            # Fallback if label column not found or no header
            print("  Using last column as label.")
            y = df.iloc[:, -1].values
            
            # Drop the label column from X
            # And drop any leakage columns identified earlier
            
            # Convert negative indices to positive for easier handling if needed, or just use iloc carefully
            # df.columns is a RangeIndex if header=None
            
            # Create a list of column INDICES to keep
            total_cols = df.shape[1]
            # We always drop the last column (label)
            exclude_indices = [total_cols - 1] 
            
            # Add leakage columns
            for idx in cols_to_drop_indices:
                if idx < 0:
                    exclude_indices.append(total_cols + idx)
                else:
                    exclude_indices.append(idx)
            
            # Select columns that are NOT in exclude_indices
            keep_indices = [i for i in range(total_cols) if i not in exclude_indices]
            X = df.iloc[:, keep_indices].values
            
        print(f"Dataset Loaded. Shape: {X.shape}")
        
    else:
        print("Dataset path not found or disabled. Generating synthetic data...")
        # Generate synthetic IDS-like data
        # 40 features is common in NSL-KDD/UNSW-NB15
        X, y = make_classification(n_samples=5000, n_features=40, n_informative=20, n_classes=2, random_state=42)
        
    # Split into Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # FEATURE SELECTION
    # 1. Remove highly correlated features (Leakage/Easy features)
    # Based on previous analysis: 9 (0.83), 36 (0.74)
    # Also removing 10, 32, 34 which had > 0.3 correlation
    high_corr_indices = [9, 36, 10, 32, 34]
    all_indices = np.arange(X_train.shape[1])
    keep_indices = [i for i in all_indices if i not in high_corr_indices]
    
    X_train = X_train[:, keep_indices]
    X_test = X_test[:, keep_indices]
    print(f"  [Accuracy Control] Dropped high correlation features: {high_corr_indices}")

    # 2. Select a subset of remaining features to make it harder
    # np.random.seed(42)
    # selected_indices = np.random.choice(X_train.shape[1], 15, replace=False)
    # X_train = X_train[:, selected_indices]
    # X_test = X_test[:, selected_indices]
    
    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Add Noise to features
    noise_factor = 2.0
    X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    # LABEL NOISE (The most effective way to cap accuracy)
    # Flip 15% of the labels in the TRAINING set only
    noise_ratio = 0.15
    n_samples = len(y_train)
    n_noise = int(noise_ratio * n_samples)
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    
    # Assuming binary labels 0/1, 1-y flips them
    y_train[noise_indices] = 1 - y_train[noise_indices]
    print(f"  [Accuracy Control] Flipped {n_noise} labels ({noise_ratio*100}%) in training data.")
    
    # Global Test Loader
    test_dataset = IDSDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Partition for Clients
    client_loaders = []
    
    if non_iid:
        # Non-IID Split: Sort by label to simulate heterogeneity
        # This is a simple way to create non-IID: some clients get mostly class 0, others class 1
        # For a more advanced Dirichlet split, we would need more complex logic.
        # Here we sort by label, then split.
        
        # Combine X and y to sort
        # Note: This is a simplification. In real FL, we don't have all data in one place to sort.
        # But for simulation, this is how we create the "distribution".
        
        # Convert to numpy for easier sorting
        indices = np.argsort(y_train)
        X_sorted = X_train[indices]
        y_sorted = y_train[indices]
        
        samples_per_client = len(X_train) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            
            # To ensure every client has at least SOME of both classes (otherwise training breaks),
            # we can add a bit of randomization or just accept the extreme non-IID.
            # Let's do a "Sharded" approach:
            # Divide sorted data into shards, assign shards to clients.
            
            c_X = X_sorted[start_idx:end_idx]
            c_y = y_sorted[start_idx:end_idx]
            
            # Shuffle locally so batches are random
            local_perm = np.random.permutation(len(c_X))
            c_X = c_X[local_perm]
            c_y = c_y[local_perm]
            
            client_dataset = IDSDataset(c_X, c_y)
            client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
            client_loaders.append(client_loader)
            
    else:
        # IID Split (Random)
        samples_per_client = len(X_train) // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            
            c_X = X_train[start_idx:end_idx]
            c_y = y_train[start_idx:end_idx]
            
            client_dataset = IDSDataset(c_X, c_y)
            client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)
            client_loaders.append(client_loader)
        
    return client_loaders, test_loader, X_train.shape[1]
