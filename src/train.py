import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import yaml
from pathlib import Path
import optuna
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Set matplotlib backend and style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

class ModelTrainer:
    def __init__(self, config_path="config/model_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create results directory
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Agriculture color scheme
        self.colors = {
            'market': '#2E8B57',      # Sea Green
            'yield': '#228B22',       # Forest Green  
            'sustainability': '#32CD32', # Lime Green
            'accent': '#FFD700',      # Gold
            'neutral': '#708090'      # Slate Gray
        }
        
        print(f"Using device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def objective(self, trial, train_loader, val_loader, input_size, model_name):
        """Optuna objective function for hyperparameter tuning"""
        
        # Model-specific hyperparameter ranges
        if "sustainability" in model_name.lower():
            # More complex architecture for sustainability
            hidden_size_1 = trial.suggest_int('hidden_size_1', 256, 512, step=64)
            hidden_size_2 = trial.suggest_int('hidden_size_2', 128, 384, step=32)
            hidden_size_3 = trial.suggest_int('hidden_size_3', 64, 192, step=16)
            hidden_size_4 = trial.suggest_int('hidden_size_4', 32, 96, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
            ensemble_size = 4  # Larger ensemble for complex task
            hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4]
        elif "yield" in model_name.lower():
            # Moderate complexity for yield prediction
            hidden_size_1 = trial.suggest_int('hidden_size_1', 192, 384, step=64)
            hidden_size_2 = trial.suggest_int('hidden_size_2', 96, 256, step=32)
            hidden_size_3 = trial.suggest_int('hidden_size_3', 48, 128, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.15, 0.4)
            learning_rate = trial.suggest_float('learning_rate', 5e-4, 8e-3, log=True)
            ensemble_size = 3
            hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
        else:
            # Default for market price
            hidden_size_1 = trial.suggest_int('hidden_size_1', 128, 512, step=64)
            hidden_size_2 = trial.suggest_int('hidden_size_2', 64, 256, step=32)
            hidden_size_3 = trial.suggest_int('hidden_size_3', 32, 128, step=16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            ensemble_size = 3
            hidden_sizes = [hidden_size_1, hidden_size_2, hidden_size_3]
        
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
        
        # Create temporary config
        temp_config = {
            'model': {
                'ensemble_size': ensemble_size,
                'hidden_sizes': hidden_sizes,
                'dropout_rate': dropout_rate,
                'batch_size': self.config['model']['batch_size'],
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            },
            'training': {
                'epochs': 75 if "sustainability" in model_name.lower() else 50,  # More epochs for complex task
                'patience': 15 if "sustainability" in model_name.lower() else 10,
                'lr_patience': 8 if "sustainability" in model_name.lower() else 5,
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
        if "sustainability" in model_name.lower() or "yield" in model_name.lower():
            # Handle variable number of hidden layers
            hidden_sizes = []
            for i in range(1, 5):  # Up to 4 hidden layers
                key = f'hidden_size_{i}'
                if key in best_params:
                    hidden_sizes.append(best_params[key])
            self.config['model']['hidden_sizes'] = hidden_sizes
        else:
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
    
    def create_training_plots(self, train_losses, val_losses, model_name):
        """Create beautiful training plots and save as images"""
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🌾 {model_name} Training Analysis 🌾', fontsize=20, fontweight='bold')
        
        epochs = list(range(1, len(train_losses) + 1))
        
        # Get model-specific color
        if 'market' in model_name.lower():
            color_main = self.colors['market']
        elif 'yield' in model_name.lower():
            color_main = self.colors['yield']
        else:
            color_main = self.colors['sustainability']
        
        # Plot 1: Loss over time
        axes[0, 0].plot(epochs, train_losses, label='Training Loss', color=color_main, linewidth=2)
        axes[0, 0].plot(epochs, val_losses, label='Validation Loss', color=self.colors['accent'], linewidth=2)
        axes[0, 0].set_title(f'{model_name} - Loss Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Loss comparison with fill
        axes[0, 1].plot(epochs, train_losses, label='Training', color=color_main, alpha=0.8)
        axes[0, 1].fill_between(epochs, train_losses, alpha=0.3, color=color_main)
        axes[0, 1].plot(epochs, val_losses, label='Validation', color=self.colors['accent'], alpha=0.8)
        axes[0, 1].fill_between(epochs, val_losses, alpha=0.3, color=self.colors['accent'])
        axes[0, 1].set_title(f'{model_name} - Loss Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Smoothed learning curve
        if len(train_losses) > 10:
            window = min(10, len(train_losses) // 5)
            smooth_train = pd.Series(train_losses).rolling(window=window).mean()
            smooth_val = pd.Series(val_losses).rolling(window=window).mean()
            
            axes[1, 0].plot(epochs, smooth_train, label='Smoothed Training', color=color_main, linewidth=3)
            axes[1, 0].plot(epochs, smooth_val, label='Smoothed Validation', color=self.colors['accent'], linewidth=3)
            axes[1, 0].set_title(f'{model_name} - Smoothed Learning Curve', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epochs')
            axes[1, 0].set_ylabel('Smoothed Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution
        axes[1, 1].hist(train_losses, alpha=0.7, label='Train Loss Distribution', color=color_main, bins=20)
        axes[1, 1].hist(val_losses, alpha=0.7, label='Val Loss Distribution', color=self.colors['accent'], bins=20)
        axes[1, 1].set_title(f'{model_name} - Loss Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Loss Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save as high-quality image
        filename = f"{model_name.lower().replace(' ', '_')}_training_analysis.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Training plot saved: {self.results_dir / filename}")
        plt.close()
    
    def create_performance_plots(self, actual, predicted, model_name):
        """Create beautiful performance plots and save as images"""
        
        # Calculate metrics
        residuals = actual - predicted
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs(residuals / (actual + 1e-6))) * 100
        
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'🎯 {model_name} Performance Analysis (R² = {r2:.4f}) 🎯', fontsize=20, fontweight='bold')
        
        # Get model-specific color
        if 'market' in model_name.lower():
            color_main = self.colors['market']
        elif 'yield' in model_name.lower():
            color_main = self.colors['yield']
        else:
            color_main = self.colors['sustainability']
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].scatter(actual, predicted, alpha=0.6, color=color_main, s=50)
        min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_title(f'{model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Actual Values')
        axes[0, 0].set_ylabel('Predicted Values')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² annotation
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
                       fontsize=12, fontweight='bold')
        
        # Plot 2: Residual plot
        axes[0, 1].scatter(predicted, residuals, alpha=0.6, color=color_main, s=50)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color=color_main, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE (%)']
        values = [r2, rmse, mae, mape]
        colors = [self.colors['market'], self.colors['yield'], self.colors['sustainability'], self.colors['accent']]
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        axes[1, 1].set_title(f'{model_name} - Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save as high-quality image
        filename = f"{model_name.lower().replace(' ', '_')}_performance_analysis.png"
        plt.savefig(self.results_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🎯 Performance plot saved: {self.results_dir / filename}")
        plt.close()
    
    def train_model(self, model, train_loader, val_loader, model_name, tune_hyperparams=None):
        """Train a model with visualization tracking"""
        
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
        train_losses = []  # Track for visualization
        val_losses = []   # Track for visualization
        
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
        
        # Create training visualization
        self.create_training_plots(train_losses, val_losses, model_name)
        
        return model, train_losses, val_losses  # Return losses for dashboard
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate model performance with visualization"""
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
        mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100
        
        print(f"\n{model_name} Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {r2*100:.2f}%")
        
        # Create performance visualization
        self.create_performance_plots(actuals, predictions, model_name)
        
        return {
            'rmse': rmse, 
            'mae': mae, 
            'r2': r2, 
            'mape': mape,
            'predictions': predictions,
            'actuals': actuals
        }