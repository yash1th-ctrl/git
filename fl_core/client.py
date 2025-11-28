import torch
import torch.nn as nn
import torch.optim as optim
import copy

class Client:
    def __init__(self, client_id, model, train_loader, device, config):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self):
        """
        Trains the local model on local data.
        Returns: state_dict of the trained model, average loss, accuracy
        """
        self.model.to(self.device)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.LR, momentum=0.9)
        
        epoch_loss = []
        correct = 0
        total = 0
        
        for epoch in range(self.config.EPOCHS_PER_CLIENT):
            batch_loss = []
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                # Phase II: DP-SGD Clipping would go here
                
                optimizer.step()
                
                batch_loss.append(loss.item())
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        accuracy = 100 * correct / total
        
        # --- Security: Authentication ---
        # Attach token to the payload
        payload = {
            "weights": copy.deepcopy(self.model.state_dict()),
            "auth_token": self.config.AUTH_TOKEN,
            "client_id": self.client_id
        }
        
        return payload, avg_loss, accuracy
