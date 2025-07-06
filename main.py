import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler

from src.data_loader import DataLoader as DataLoaderClass
from src.preprocessing import DataPreprocessor
from src.models import EnsembleModel, AgriculturalDataset
from src.train import ModelTrainer

def main():
    # Create directories
    Path("saved_models").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    # Load configuration
    with open("config/model_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Initialize components
    data_loader = DataLoaderClass()
    preprocessor = DataPreprocessor()
    trainer = ModelTrainer()
    
    print("Loading and preprocessing data...")
    
    # Load data
    market_df = data_loader.load_market_data()
    farmer_df = data_loader.load_farmer_data()
    target_cols = data_loader.get_target_columns()
    
    # Preprocess data
    market_df = preprocessor.preprocess_market_data(market_df)
    farmer_df = preprocessor.preprocess_farmer_data(farmer_df)
    
    print(f"Market data shape: {market_df.shape}")
    print(f"Farmer data shape: {farmer_df.shape}")
    
    # Train Market Price Model
    print("\n" + "="*60)
    print("TRAINING MARKET PRICE PREDICTION MODEL")
    print("="*60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data_for_training(
        market_df, target_cols['market_target'], model_name="market_price"
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets and loaders
    train_dataset = AgriculturalDataset(X_train_scaled, y_train)
    val_dataset = AgriculturalDataset(X_val_scaled, y_val)
    test_dataset = AgriculturalDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    
    # Create and train model
    market_model = EnsembleModel(X_train_scaled.shape[1], config)
    market_model = trainer.train_model(market_model, train_loader, val_loader, "market_price")
    market_metrics = trainer.evaluate_model(market_model, test_loader, "Market Price")
    
    # Train Crop Yield Model
    print("\n" + "="*60)
    print("TRAINING CROP YIELD PREDICTION MODEL")
    print("="*60)
    
    farmer_yield_df = farmer_df.copy()
    X_train_f, X_val_f, X_test_f, y_train_f, y_val_f, y_test_f = preprocessor.prepare_data_for_training(
        farmer_yield_df, target_cols['yield_target'], model_name="crop_yield"
    )
    
    # Scale features
    scaler_f = RobustScaler()
    X_train_f_scaled = scaler_f.fit_transform(X_train_f)
    X_val_f_scaled = scaler_f.transform(X_val_f)
    X_test_f_scaled = scaler_f.transform(X_test_f)
    
    # Create datasets and loaders
    train_dataset_f = AgriculturalDataset(X_train_f_scaled, y_train_f)
    val_dataset_f = AgriculturalDataset(X_val_f_scaled, y_val_f)
    test_dataset_f = AgriculturalDataset(X_test_f_scaled, y_test_f)
    
    train_loader_f = DataLoader(train_dataset_f, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader_f = DataLoader(val_dataset_f, batch_size=config['model']['batch_size'], shuffle=False)
    test_loader_f = DataLoader(test_dataset_f, batch_size=config['model']['batch_size'], shuffle=False)
    
    # Train yield model
    yield_model = EnsembleModel(X_train_f_scaled.shape[1], config)
    yield_model = trainer.train_model(yield_model, train_loader_f, val_loader_f, "crop_yield")
    yield_metrics = trainer.evaluate_model(yield_model, test_loader_f, "Crop Yield")
    
    # Train Sustainability Model
    print("\n" + "="*60)
    print("TRAINING SUSTAINABILITY PREDICTION MODEL")
    print("="*60)
    
    sustainability_df = farmer_df.copy()
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s = preprocessor.prepare_data_for_training(
        sustainability_df, target_cols['sustainability_target'], model_name="sustainability"
    )
    
    # Scale features
    scaler_s = RobustScaler()
    X_train_s_scaled = scaler_s.fit_transform(X_train_s)
    X_val_s_scaled = scaler_s.transform(X_val_s)
    X_test_s_scaled = scaler_s.transform(X_test_s)
    
    # Create datasets and loaders
    train_dataset_s = AgriculturalDataset(X_train_s_scaled, y_train_s)
    val_dataset_s = AgriculturalDataset(X_val_s_scaled, y_val_s)
    test_dataset_s = AgriculturalDataset(X_test_s_scaled, y_test_s)
    
    train_loader_s = DataLoader(train_dataset_s, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader_s = DataLoader(val_dataset_s, batch_size=config['model']['batch_size'], shuffle=False)
    test_loader_s = DataLoader(test_dataset_s, batch_size=config['model']['batch_size'], shuffle=False)
    
    # Train sustainability model
    sustainability_model = EnsembleModel(X_train_s_scaled.shape[1], config)
    sustainability_model = trainer.train_model(sustainability_model, train_loader_s, val_loader_s, "sustainability", tune_hyperparams=True)
    sustainability_metrics = trainer.evaluate_model(sustainability_model, test_loader_s, "Sustainability")
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Market Price Model:")
    print(f"  - R² Score: {market_metrics['r2']:.4f} ({market_metrics['r2']*100:.2f}%)")
    print(f"  - RMSE: {market_metrics['rmse']:.4f}")
    print(f"  - MAPE: {market_metrics['mape']:.2f}%")
    
    print(f"\nCrop Yield Model:")
    print(f"  - R² Score: {yield_metrics['r2']:.4f} ({yield_metrics['r2']*100:.2f}%)")
    print(f"  - RMSE: {yield_metrics['rmse']:.4f}")
    print(f"  - MAPE: {yield_metrics['mape']:.2f}%")
    
    print(f"\nSustainability Model:")
    print(f"  - R² Score: {sustainability_metrics['r2']:.4f} ({sustainability_metrics['r2']*100:.2f}%)")
    print(f"  - RMSE: {sustainability_metrics['rmse']:.4f}")
    print(f"  - MAPE: {sustainability_metrics['mape']:.2f}%")
    
    avg_r2 = (market_metrics['r2'] + yield_metrics['r2'] + sustainability_metrics['r2']) / 3
    print(f"\nAverage R² Score across all models: {avg_r2:.4f} ({avg_r2*100:.2f}%)")
    print(f"Total models created: 3")
    print(f"Models saved in: saved_models/")
    
    # Performance improvement suggestions
    if avg_r2 < 0.8:
        print("\n" + "="*60)
        print("PERFORMANCE IMPROVEMENT SUGGESTIONS")
        print("="*60)
        print("1. Increase training data size")
        print("2. Try different feature engineering approaches")
        print("3. Experiment with different model architectures")
        print("4. Consider using cross-validation for better evaluation")
        print("5. Add more domain-specific features")

if __name__ == "__main__":
    main()
