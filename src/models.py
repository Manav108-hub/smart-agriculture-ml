import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgriculturalDataset(Dataset):
    def __init__(self, X, y):
        # Convert pandas Series/DataFrame to numpy arrays if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.2):
        super(ImprovedNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

class EnsembleModel(nn.Module):
    def __init__(self, input_size, config):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([
            ImprovedNeuralNetwork(
                input_size, 
                config['model']['hidden_sizes'], 
                config['model']['dropout_rate']
            ) for _ in range(config['model']['ensemble_size'])
        ])
        
        # Add a meta-learner for ensemble combination
        self.meta_learner = nn.Linear(config['model']['ensemble_size'], 1)
        
    def forward(self, x):
        predictions = torch.stack([model(x) for model in self.models], dim=2)
        # Use meta-learner to combine predictions
        weights = F.softmax(self.meta_learner.weight, dim=1)
        ensemble_pred = torch.sum(predictions * weights, dim=2)
        return ensemble_pred