import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import yaml
from pathlib import Path
import optuna
from typing import Dict, Any

class ModelTrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def objective(self, trial, train_loader, val_loader, input_size, model_name):
        """Optuna objective function for hyperparameter tuning"""
        
        # Suggest hyperparameters
        hidden_size_1 = trial.suggest_int('hidden_size_1', 128, 512, step=64)
        hidden_size_2 = trial.suggest_int('hidden_size_2', 64, 256, step=32)
        hidden_size_3 = trial.suggest_int('hidden_size_3', 32, 128, step=16)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        
        # Create temporary config
        temp_config = {
            'model': {
                'ensemble_size': 2,  # Reduced for faster tuning
                'hidden_sizes': [hidden_size_1, hidden_size_2, hidden_size_3],
                'dropout_rate': dropout_rate,
                'batch_size': self.config['model']['batch_size'],
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'training': {
                'epochs': 50,  # Reduced for faster tuning
                'patience': 10,
                'lr_patience': 5,
                'lr_factor': 0.7,
                'grad_clip': 0.5
            }
        }
        
        # Import here to avoid circular imports
        from src.models import EnsembleModel
        
        # Create and train model
        model = EnsembleModel(input_size, temp_config)
        model = model.to(self.device)
        
        # Training setup
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.7
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(temp_config['training']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), temp_config['training']['grad_clip'])
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= temp_config['training']['patience']:
                    break
        
        # Calculate R² score for validation set
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        r2 = r2_score(actuals, predictions)
        
        return r2
    
    def tune_hyperparameters(self, train_loader, val_loader, input_size, model_name, n_trials=None):
        """Run hyperparameter tuning using Optuna"""
        if n_trials is None:
            n_trials = self.config.get('training', {}).get('tuning_trials', 20)
            
        print(f"Starting hyperparameter tuning for {model_name} ({n_trials} trials)...")
        
        study = optuna.create_study(direction='maximize', 
                                  study_name=f'{model_name}_tuning',
                                  pruner=optuna.pruners.MedianPruner())
        
        objective_with_args = lambda trial: self.objective(trial, train_loader, val_loader, input_size, model_name)
        
        study.optimize(objective_with_args, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest hyperparameters for {model_name}:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print(f"Best R² score: {study.best_value:.4f}")
        
        # Update config with best parameters
        best_params = study.best_params
        self.config['model']['hidden_sizes'] = [
            best_params['hidden_size_1'],
            best_params['hidden_size_2'],
            best_params['hidden_size_3']
        ]
        self.config['model']['dropout_rate'] = best_params['dropout_rate']
        self.config['model']['learning_rate'] = best_params['learning_rate']
        self.config['model']['weight_decay'] = best_params['weight_decay']
        
        # Also update training config
        self.config['training']['learning_rate'] = best_params['learning_rate']
        self.config['training']['weight_decay'] = best_params['weight_decay']
        
        return study.best_params
    
    def train_model(self, model, train_loader, val_loader, model_name, tune_hyperparams=None):
        """Train a model with optional hyperparameter tuning"""
        
        # Use config setting if not explicitly provided
        if tune_hyperparams is None:
            tune_hyperparams = self.config.get('training', {}).get('tune_hyperparams', True)
        
        # Hyperparameter tuning
        if tune_hyperparams:
            input_size = next(iter(train_loader))[0].shape[1]
            best_params = self.tune_hyperparameters(train_loader, val_loader, input_size, model_name)
            
            # Recreate model with best hyperparameters
            from src.models import EnsembleModel
            model = EnsembleModel(input_size, self.config)
        else:
            print(f"Skipping hyperparameter tuning for {model_name}")
        
        model = model.to(self.device)
        
        # Use Huber loss for robustness
        criterion = nn.HuberLoss(delta=1.0)
        
        # Use different optimizers for different components
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config['model'].get('learning_rate', self.config['training']['learning_rate']),
            weight_decay=self.config['model'].get('weight_decay', self.config['training']['weight_decay']),
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"Training {model_name} with optimized hyperparameters...")
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['grad_clip'])
                
                optimizer.step()
                train_loss += loss.item()
                train_batches += 1
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            
            train_loss /= train_batches
            val_loss /= val_batches
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            # Progress reporting
            if epoch % 50 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Early stopping with improved patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'saved_models/{model_name}_best.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        model.load_state_dict(torch.load(f'saved_models/{model_name}_best.pth'))
        print(f"Best validation loss: {best_val_loss:.4f}")
        return model
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate model performance with additional metrics"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100
        
        print(f"\n{model_name} Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {r2*100:.2f}%")
        
        return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}