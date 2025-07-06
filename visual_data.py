"""
Utility script to visualize model performance and results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_model_performance(actual, predicted, model_name):
    """Plot actual vs predicted values"""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actual, predicted, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Actual vs Predicted')
    
    # Residual plot
    plt.subplot(1, 2, 2)
    residuals = actual - predicted
    plt.scatter(predicted, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name}: Residual Plot')
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_names, importance_scores, model_name):
    """Plot feature importance"""
    plt.figure(figsize=(10, 8))
    
    # Sort features by importance
    sorted_idx = np.argsort(importance_scores)[::-1][:15]  # Top 15 features
    
    plt.barh(range(len(sorted_idx)), importance_scores[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(f'{model_name}: Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_features.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_dashboard(market_metrics, yield_metrics, sustainability_metrics):
    """Create a comprehensive performance dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² scores comparison
    models = ['Market Price', 'Crop Yield', 'Sustainability']
    r2_scores = [market_metrics['r2'], yield_metrics['r2'], sustainability_metrics['r2']]
    
    axes[0, 0].bar(models, r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('Model R² Scores Comparison')
    axes[0, 0].set_ylim(0, 1)
    
    # RMSE comparison
    rmse_scores = [market_metrics['rmse'], yield_metrics['rmse'], sustainability_metrics['rmse']]
    axes[0, 1].bar(models, rmse_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('Model RMSE Comparison')
    
    # MAPE comparison
    mape_scores = [market_metrics['mape'], yield_metrics['mape'], sustainability_metrics['mape']]
    axes[1, 0].bar(models, mape_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_title('Model MAPE Comparison')
    
    # Overall performance radar chart
    metrics = ['R²', 'RMSE (inv)', 'MAPE (inv)']
    
    # Normalize metrics for radar chart (higher is better)
    market_norm = [market_metrics['r2'], 1/(1+market_metrics['rmse']), 1/(1+market_metrics['mape']/100)]
    yield_norm = [yield_metrics['r2'], 1/(1+yield_metrics['rmse']), 1/(1+yield_metrics['mape']/100)]
    sustain_norm = [sustainability_metrics['r2'], 1/(1+sustainability_metrics['rmse']), 1/(1+sustainability_metrics['mape']/100)]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    axes[1, 1] = plt.subplot(2, 2, 4, projection='polar')
    
    # Plot each model
    for name, values, color in [('Market', market_norm, 'skyblue'), 
                               ('Yield', yield_norm, 'lightgreen'), 
                               ('Sustainability', sustain_norm, 'lightcoral')]:
        values += values[:1]  # Complete the circle
        axes[1, 1].plot(angles, values, 'o-', linewidth=2, label=name, color=color)
        axes[1, 1].fill(angles, values, alpha=0.25, color=color)
    
    axes[1, 1].set_xticks(angles[:-1])
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_title('Overall Performance Comparison')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
