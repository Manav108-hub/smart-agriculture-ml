import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import DataLoader as DataLoaderClass
from src.preprocessing import DataPreprocessor
from src.models import EnsembleModel, AgriculturalDataset
from src.train import ModelTrainer

def create_final_dashboard(all_results):
    """Create a beautiful final dashboard and save as image"""
    
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    models = list(all_results.keys())
    colors = ['#2E8B57', '#228B22', '#32CD32']  # Agriculture theme colors
    
    # Create comprehensive dashboard
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('🌾 Agriculture ML Models - Comprehensive Performance Dashboard 🌾', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Extract metrics
    r2_scores = [all_results[model]['metrics']['r2'] for model in models]
    rmse_scores = [all_results[model]['metrics']['rmse'] for model in models]
    mape_scores = [all_results[model]['metrics']['mape'] for model in models]
    
    # Row 1: Metric comparisons with beautiful bars
    # R² Score Comparison
    bars1 = axes[0, 0].bar(models, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 0].set_title('🏆 R² Score Comparison', fontsize=16, fontweight='bold')
    axes[0, 0].set_ylabel('R² Score', fontweight='bold')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars1, r2_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # RMSE Comparison
    bars2 = axes[0, 1].bar(models, rmse_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 1].set_title('📉 RMSE Comparison', fontsize=16, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars2, rmse_scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # MAPE Comparison
    bars3 = axes[0, 2].bar(models, mape_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 2].set_title('📊 MAPE Comparison (%)', fontsize=16, fontweight='bold')
    axes[0, 2].set_ylabel('MAPE (%)', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars3, mape_scores):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{score:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Row 2: Training evolution and final metrics
    # Training Loss Evolution
    for i, model in enumerate(models):
        if 'train_losses' in all_results[model]:
            epochs = list(range(1, len(all_results[model]['train_losses']) + 1))
            axes[1, 0].plot(epochs, all_results[model]['train_losses'], 
                           label=f'{model}', color=colors[i], linewidth=2)
    
    axes[1, 0].set_title('📈 Training Loss Evolution', fontsize=16, fontweight='bold')
    axes[1, 0].set_xlabel('Epochs', fontweight='bold')
    axes[1, 0].set_ylabel('Training Loss', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation Loss Evolution
    for i, model in enumerate(models):
        if 'val_losses' in all_results[model]:
            epochs = list(range(1, len(all_results[model]['val_losses']) + 1))
            axes[1, 1].plot(epochs, all_results[model]['val_losses'], 
                           label=f'{model}', color=colors[i], linewidth=2, linestyle='--')
    
    axes[1, 1].set_title('✅ Validation Loss Evolution', fontsize=16, fontweight='bold')
    axes[1, 1].set_xlabel('Epochs', fontweight='bold')
    axes[1, 1].set_ylabel('Validation Loss', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Final Accuracy Comparison
    accuracy_scores = [score * 100 for score in r2_scores]
    bars4 = axes[1, 2].bar(models, accuracy_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    axes[1, 2].set_title('🎯 Final Accuracy (%)', fontsize=16, fontweight='bold')
    axes[1, 2].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[1, 2].set_ylim(0, 100)
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars4, accuracy_scores):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add performance summary text box
    avg_r2 = sum(r2_scores) / len(r2_scores)
    best_model = models[r2_scores.index(max(r2_scores))]
    worst_model = models[r2_scores.index(min(r2_scores))]
    
    summary_text = f"""🏆 PERFORMANCE SUMMARY 🏆
Best Model: {best_model} ({max(r2_scores):.3f})
Worst Model: {worst_model} ({min(r2_scores):.3f})
Average R²: {avg_r2:.3f}
Overall Performance: {avg_r2*100:.1f}%

📊 Model Rankings:"""
    
    # Add rankings
    sorted_models = sorted(zip(models, r2_scores), key=lambda x: x[1], reverse=True)
    for i, (model, score) in enumerate(sorted_models, 1):
        summary_text += f"\n{i}. {model}: {score:.3f}"
    
    # Add text box to the figure
    fig.text(0.02, 0.02, summary_text, fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for summary text
    
    # Save final dashboard
    dashboard_filename = "🌾_FINAL_AGRICULTURE_DASHBOARD.png"
    plt.savefig(results_dir / dashboard_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🌟 Final dashboard saved: {results_dir / dashboard_filename}")
    plt.close()
    
    # Create individual model comparison chart
    create_model_comparison_chart(all_results, results_dir)

def create_model_comparison_chart(all_results, results_dir):
    """Create a detailed model comparison chart"""
    
    models = list(all_results.keys())
    metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
    
    # Prepare data for comparison
    comparison_data = []
    for model in models:
        comparison_data.append([
            all_results[model]['metrics']['r2'],
            all_results[model]['metrics']['rmse'],
            all_results[model]['metrics']['mae'],
            all_results[model]['metrics']['mape']
        ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize data for better visualization (except R² which is already 0-1)
    normalized_data = np.array(comparison_data).T
    normalized_data[1] = normalized_data[1] / np.max(normalized_data[1])  # Normalize RMSE
    normalized_data[2] = normalized_data[2] / np.max(normalized_data[2])  # Normalize MAE
    normalized_data[3] = normalized_data[3] / 100  # MAPE to 0-1 scale
    
    # Create heatmap
    im = ax.imshow(normalized_data, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(metrics)))
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_yticklabels(metrics, fontweight='bold')
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(models)):
            if i == 0:  # R²
                text = f'{comparison_data[j][i]:.3f}'
            elif i == 3:  # MAPE
                text = f'{comparison_data[j][i]:.1f}%'
            else:  # RMSE, MAE
                text = f'{comparison_data[j][i]:.3f}'
            
            ax.text(j, i, text, ha="center", va="center", fontweight='bold', fontsize=12)
    
    ax.set_title('🔍 Detailed Model Performance Heatmap 🔍', fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Performance (Green = Better)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison_heatmap.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🔍 Comparison heatmap saved: {results_dir}/model_comparison_heatmap.png")
    plt.close()

def create_prediction_samples_plot(all_results, results_dir):
    """Create a plot showing sample predictions vs actual values"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('🎯 Sample Predictions vs Actual Values 🎯', fontsize=20, fontweight='bold')
    
    colors = ['#2E8B57', '#228B22', '#32CD32']
    
    for i, (model_name, results) in enumerate(all_results.items()):
        actual = results['metrics']['actuals'][:50]  # First 50 samples
        predicted = results['metrics']['predictions'][:50]
        
        # Sample indices
        sample_indices = range(len(actual))
        
        axes[i].plot(sample_indices, actual, 'o-', label='Actual', color='black', linewidth=2, markersize=6)
        axes[i].plot(sample_indices, predicted, 's-', label='Predicted', color=colors[i], linewidth=2, markersize=6)
        
        axes[i].set_title(f'{model_name}\nR² = {results["metrics"]["r2"]:.3f}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Sample Index', fontweight='bold')
        axes[i].set_ylabel('Value', fontweight='bold')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Add RMSE annotation
        rmse = results['metrics']['rmse']
        axes[i].text(0.02, 0.98, f'RMSE: {rmse:.3f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                    fontsize=11, fontweight='bold', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(results_dir / "sample_predictions_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    print(f"🎯 Sample predictions plot saved: {results_dir}/sample_predictions_comparison.png")
    plt.close()

def main():
    # Create all required directories
    Path("saved_models").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    print("🌾 Starting Agriculture ML Training with Integrated Visualization 🌾")
    print("=" * 80)
    print(f"📁 Created directories: saved_models/, config/, data/, results/")
    print("=" * 80)
    
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
    
    # Store all results for final dashboard
    all_results = {}
    
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
    market_model, market_train_losses, market_val_losses = trainer.train_model(
        market_model, train_loader, val_loader, "Market Price"
    )
    market_metrics = trainer.evaluate_model(market_model, test_loader, "Market Price")
    
    all_results['Market Price'] = {
        'metrics': market_metrics,
        'train_losses': market_train_losses,
        'val_losses': market_val_losses
    }
    
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
    yield_model, yield_train_losses, yield_val_losses = trainer.train_model(
        yield_model, train_loader_f, val_loader_f, "Crop Yield"
    )
    yield_metrics = trainer.evaluate_model(yield_model, test_loader_f, "Crop Yield")
    
    all_results['Crop Yield'] = {
        'metrics': yield_metrics,
        'train_losses': yield_train_losses,
        'val_losses': yield_val_losses
    }
    
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
    sustainability_model, sustain_train_losses, sustain_val_losses = trainer.train_model(
        sustainability_model, train_loader_s, val_loader_s, "Sustainability"
    )
    sustainability_metrics = trainer.evaluate_model(sustainability_model, test_loader_s, "Sustainability")
    
    all_results['Sustainability'] = {
        'metrics': sustainability_metrics,
        'train_losses': sustain_train_losses,
        'val_losses': sustain_val_losses
    }
    
    # Create Final Comprehensive Dashboard
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATION DASHBOARD")
    print("="*60)
    
    create_final_dashboard(all_results)
    create_prediction_samples_plot(all_results, Path("results"))
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"Market Price Model:")
    print(f"  - R² Score: {all_results['Market Price']['metrics']['r2']:.4f} ({all_results['Market Price']['metrics']['r2']*100:.2f}%)")
    print(f"  - RMSE: {all_results['Market Price']['metrics']['rmse']:.4f}")
    print(f"  - MAPE: {all_results['Market Price']['metrics']['mape']:.2f}%")
    
    print(f"\nCrop Yield Model:")
    print(f"  - R² Score: {all_results['Crop Yield']['metrics']['r2']:.4f} ({all_results['Crop Yield']['metrics']['r2']*100:.2f}%)")
    print(f"  - RMSE: {all_results['Crop Yield']['metrics']['rmse']:.4f}")
    print(f"  - MAPE: {all_results['Crop Yield']['metrics']['mape']:.2f}%")
    
    print(f"\nSustainability Model:")
    print(f"  - R² Score: {all_results['Sustainability']['metrics']['r2']:.4f} ({all_results['Sustainability']['metrics']['r2']*100:.2f}%)")
    print(f"  - RMSE: {all_results['Sustainability']['metrics']['rmse']:.4f}")
    print(f"  - MAPE: {all_results['Sustainability']['metrics']['mape']:.2f}%")
    
    avg_r2 = sum([all_results[model]['metrics']['r2'] for model in all_results.keys()]) / len(all_results)
    print(f"\nAverage R² Score across all models: {avg_r2:.4f} ({avg_r2*100:.2f}%)")
    print(f"Total models created: {len(all_results)}")
    print(f"Models saved in: saved_models/")
    
    print("\n" + "🎨" * 20)
    print("🎨 BEAUTIFUL VISUALIZATIONS CREATED! 🎨")
    print("🎨" * 20)
    print(f"📊 All visualization images saved in: results/")
    print(f"🌟 Check the following files:")
    print(f"   - results/🌾_FINAL_AGRICULTURE_DASHBOARD.png")
    print(f"   - results/model_comparison_heatmap.png")
    print(f"   - results/sample_predictions_comparison.png")
    print(f"   - results/market_price_training_analysis.png")
    print(f"   - results/market_price_performance_analysis.png")
    print(f"   - results/crop_yield_training_analysis.png")
    print(f"   - results/crop_yield_performance_analysis.png")
    print(f"   - results/sustainability_training_analysis.png")
    print(f"   - results/sustainability_performance_analysis.png")
    print("🎨" * 20)
    
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