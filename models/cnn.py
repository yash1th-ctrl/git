import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        # Input is expected to be (Batch, 1, Features) or (Batch, Channels, Height, Width)
        # For IDS tabular data, we often reshape 1D features into a pseudo-image or use 1D Conv.
        # Assuming 1D Conv for feature vector.
        
        # Reduced capacity to prevent 100% accuracy (Overfitting/Leakage)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5) # Strong dropout
        
        # Calculate size after convolutions and pooling
        # If input_dim is N, 
        # conv1 -> N
        # pool -> N/2
        self.flatten_dim = 4 * (input_dim // 2) 
        
        self.fc1 = nn.Linear(self.flatten_dim, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, input_dim] -> need [batch_size, 1, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # No Softmax here, CrossEntropyLoss handles it
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        
        # LSTM expects (Batch, Seq_Len, Features)
        # After conv/pool: (Batch, 16, Dim/2) -> Permute to (Batch, Dim/2, 16) for LSTM?
        # Or treat Conv output channels as features for LSTM time steps?
        # Usually for IDS: Conv extracts spatial features, LSTM temporal.
        # Let's keep it simple: Conv -> LSTM -> FC
        
        self.flatten_dim = input_dim // 2
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        # x: (Batch, 16, Dim/2)
        # Permute for LSTM: (Batch, Dim/2, 16) -> Sequence length is Dim/2
        x = x.permute(0, 2, 1) 
        
        _, (hn, _) = self.lstm(x)
        # hn: (1, Batch, 32)
        x = hn[-1]
        x = self.fc(x)
        return x
