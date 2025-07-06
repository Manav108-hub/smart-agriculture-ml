import pandas as pd
import numpy as np
import yaml
from pathlib import Path

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
            print(f"Market data columns: {list(df.columns)}")
            print(f"Market data dtypes:\n{df.dtypes}")
            
            # Display first few rows for verification
            print(f"First 3 rows of market data:")
            print(df.head(3))
            
            return df
        except FileNotFoundError:
            print(f"Market data file not found at {market_path}")
            print("Please ensure the file exists or check the path in config/data_config.yaml")
            raise
    
    def load_farmer_data(self):
        """Load farmer advice data"""
        farmer_path = self.config['data']['farmer_file']
        try:
            print(f"Loading farmer data from {farmer_path}")
            df = pd.read_csv(farmer_path)
            print(f"Farmer data shape: {df.shape}")
            print(f"Farmer data columns: {list(df.columns)}")
            print(f"Farmer data dtypes:\n{df.dtypes}")
            
            # Display first few rows for verification
            print(f"First 3 rows of farmer data:")
            print(df.head(3))
            
            return df
        except FileNotFoundError:
            print(f"Farmer data file not found at {farmer_path}")
            print("Please ensure the file exists or check the path in config/data_config.yaml")
            raise
    
    def get_target_columns(self):
        """Get target column names"""
        return {
            'market_target': self.config['data']['market_target'],
            'yield_target': self.config['data']['farmer_yield_target'],
            'sustainability_target': self.config['data']['farmer_sustainability_target']
        }