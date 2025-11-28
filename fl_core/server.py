import torch
import copy
import numpy as np

class Server:
    def __init__(self, global_model, test_loader, device, config):
        self.global_model = global_model
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
    def aggregate(self, client_weights):
        """
        FedAvg Aggregation.
        """
        # Initialize global weights with zero
        global_dict = copy.deepcopy(client_weights[0])
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k], dtype=torch.float32)
            
        # Sum all weights
        # TODO: Weighted average based on number of samples per client
        for w in client_weights:
            for k in global_dict.keys():
                global_dict[k] += w[k]
                
        # Average
        for k in global_dict.keys():
            global_dict[k] = torch.div(global_dict[k], len(client_weights))
            
        self.global_model.load_state_dict(global_dict)
        
    def evaluate(self):
        """
        Evaluate the global model on the test set.
        """
        self.global_model.to(self.device)
        self.global_model.eval()
        
        correct = 0
        total = 0
        loss = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
        accuracy = 100 * correct / total
        avg_loss = loss / len(self.test_loader)
        
        from sklearn.metrics import f1_score, precision_score, recall_score
        f1 = f1_score(all_labels, all_preds, average='weighted')
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return accuracy, avg_loss, f1, precision, recall
