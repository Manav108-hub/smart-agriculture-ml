import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.datasets import make_regression

class DataLoader:
    def __init__(self, config_path="config/data_config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
    def load_market_data(self):
        """Load market research data"""
        market_path = self.config['data']['market_file']
        try:
            print(f"Loading market data from {market_path}")
            df = pd.read_csv(market_path)
            print(f"Market data shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Market data file not found. Creating synthetic data...")
            return self._create_synthetic_market_data()
    
    def load_farmer_data(self):
        """Load farmer advice data"""
        farmer_path = self.config['data']['farmer_file']
        try:
            print(f"Loading farmer data from {farmer_path}")
            df = pd.read_csv(farmer_path)
            print(f"Farmer data shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Farmer data file not found. Creating synthetic data...")
            return self._create_synthetic_farmer_data()
    
    def _create_synthetic_market_data(self, n_samples=2000):
        """Create synthetic market data for testing"""
        np.random.seed(42)
        
        # Generate base features
        demand_index = np.random.normal(50, 15, n_samples)
        supply_index = np.random.normal(45, 12, n_samples)
        economic_indicator = np.random.normal(0.5, 0.2, n_samples)
        weather_impact = np.random.normal(0.7, 0.3, n_samples)
        seasonal_factor = np.random.normal(1.0, 0.2, n_samples)
        consumer_trend = np.random.normal(60, 20, n_samples)
        
        # Create realistic price relationship
        base_price = 500 + demand_index * 2 - supply_index * 1.5
        price_noise = np.random.normal(0, 50, n_samples)
        market_price = base_price + economic_indicator * 100 + weather_impact * 80 + price_noise
        
        # Competitor price follows similar pattern with offset
        competitor_price = market_price + np.random.normal(-50, 30, n_samples)
        
        df = pd.DataFrame({
            'Market_Price_per_ton': market_price,
            'Demand_Index': demand_index,
            'Supply_Index': supply_index,
            'Competitor_Price_per_ton': competitor_price,
            'Economic_Indicator': economic_indicator,
            'Weather_Impact_Score': weather_impact,
            'Seasonal_Factor': seasonal_factor,
            'Consumer_Trend_Index': consumer_trend
        })
        
        return df
    
    def _create_synthetic_farmer_data(self, n_samples=2000):
        """Create synthetic farmer data for testing"""
        np.random.seed(42)
        
        # Generate base features
        soil_ph = np.random.normal(6.5, 0.8, n_samples)
        soil_moisture = np.random.normal(0.3, 0.1, n_samples)
        temperature = np.random.normal(25, 5, n_samples)
        rainfall = np.random.normal(800, 200, n_samples)
        fertilizer_usage = np.random.normal(150, 50, n_samples)
        pesticide_usage = np.random.normal(20, 10, n_samples)
        
        # Create crop types
        crop_types = ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Barley']
        crop_type = np.random.choice(crop_types, n_samples)
        
        # Create realistic yield relationship
        base_yield = 3.0 + (soil_ph - 5.5) * 0.5 + soil_moisture * 5
        base_yield += (temperature - 20) * 0.1 + rainfall * 0.002
        base_yield += fertilizer_usage * 0.01 - pesticide_usage * 0.05
        base_yield += np.random.normal(0, 0.5, n_samples)
        base_yield = np.maximum(base_yield, 0.5)  # Ensure positive yield
        
        # Create sustainability score
        sustainability = 80 - pesticide_usage * 0.5 - fertilizer_usage * 0.1
        sustainability += soil_ph * 2 + soil_moisture * 10
        sustainability += np.random.normal(0, 5, n_samples)
        sustainability = np.clip(sustainability, 0, 100)
        
        df = pd.DataFrame({
            'Soil_pH': soil_ph,
            'Soil_Moisture': soil_moisture,
            'Temperature_C': temperature,
            'Rainfall_mm': rainfall,
            'Crop_Type': crop_type,
            'Fertilizer_Usage_kg': fertilizer_usage,
            'Pesticide_Usage_kg': pesticide_usage,
            'Crop_Yield_ton': base_yield,
            'Sustainability_Score': sustainability
        })
        
        return df
    
    def get_target_columns(self):
        """Get target column names"""
        return {
            'market_target': self.config['data']['market_target'],
            'yield_target': self.config['data']['farmer_yield_target'],
            'sustainability_target': self.config['data']['farmer_sustainability_target']
        }