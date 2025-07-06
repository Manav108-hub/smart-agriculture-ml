import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

class DataPreprocessor:
    def __init__(self):
        self.market_scaler = RobustScaler()
        self.farmer_scaler = RobustScaler()
        self.label_encoders = {}
        self.feature_selectors = {}
    
    def clean_data(self, df):
        """Clean data by handling outliers and missing values"""
        df = df.copy()
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove outliers using IQR method
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
        return df
    
    def preprocess_market_data(self, df):
        """Preprocess market data with improved feature engineering"""
        print("Preprocessing market data...")
        df = self.clean_data(df)
        
        # Advanced feature engineering
        df['Price_Demand_Ratio'] = df['Market_Price_per_ton'] / (df['Demand_Index'] + 1e-6)
        df['Supply_Demand_Ratio'] = df['Supply_Index'] / (df['Demand_Index'] + 1e-6)
        df['Competitor_Price_Diff'] = df['Market_Price_per_ton'] - df['Competitor_Price_per_ton']
        df['Economic_Weather_Int'] = df['Economic_Indicator'] * df['Weather_Impact_Score']
        df['Market_Pressure'] = (df['Demand_Index'] - df['Supply_Index']) / (df['Demand_Index'] + df['Supply_Index'])
        df['Price_Volatility'] = df['Market_Price_per_ton'] / df['Competitor_Price_per_ton']
        df['Seasonal_Demand'] = df['Seasonal_Factor'] * df['Consumer_Trend_Index']
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def preprocess_farmer_data(self, df):
        """Preprocess farmer data with improved feature engineering"""
        print("Preprocessing farmer data...")
        df = self.clean_data(df)
        
        # Advanced feature engineering
        df['Fertilizer_Efficiency'] = df['Crop_Yield_ton'] / (df['Fertilizer_Usage_kg'] + 1e-6)
        df['Pesticide_Efficiency'] = df['Crop_Yield_ton'] / (df['Pesticide_Usage_kg'] + 1e-6)
        df['Soil_Health_Index'] = df['Soil_pH'] * df['Soil_Moisture']
        df['Weather_Condition'] = df['Temperature_C'] * df['Rainfall_mm'] / 1000
        df['Input_Ratio'] = df['Fertilizer_Usage_kg'] / (df['Pesticide_Usage_kg'] + 1e-6)
        df['Optimal_pH'] = np.abs(df['Soil_pH'] - 6.5)  # Distance from optimal pH
        df['Moisture_Temp_Int'] = df['Soil_Moisture'] * df['Temperature_C']
        df['Rainfall_Temp_Ratio'] = df['Rainfall_mm'] / (df['Temperature_C'] + 1e-6)
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
        
        return df
    
    def select_features(self, X, y, k=20, model_name=""):
        """Select top k features"""
        if model_name not in self.feature_selectors:
            self.feature_selectors[model_name] = SelectKBest(f_regression, k=min(k, X.shape[1]))
            X_selected = self.feature_selectors[model_name].fit_transform(X, y)
        else:
            X_selected = self.feature_selectors[model_name].transform(X)
        
        selected_features = self.feature_selectors[model_name].get_support(indices=True)
        print(f"Selected {len(selected_features)} features for {model_name}")
        return X_selected, selected_features
    
    def prepare_data_for_training(self, df, target_col, test_size=0.2, val_size=0.2, model_name=""):
        """Prepare data for model training with feature selection"""
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Split data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=None
        )
        
        # Feature selection on training data
        X_train_selected, selected_features = self.select_features(X_train, y_train, model_name=model_name)
        X_val_selected = self.feature_selectors[model_name].transform(X_val)
        X_test_selected = self.feature_selectors[model_name].transform(X_test)
        
        return X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test
