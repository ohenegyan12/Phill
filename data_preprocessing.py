#!/usr/bin/env python3
"""
Data Preprocessing Module for Student Performance Prediction

This module handles:
1. Data cleaning and validation
2. Categorical variable encoding
3. Feature scaling and normalization
4. Train-test splitting
5. Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles comprehensive data preprocessing for student performance prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = None
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.is_fitted = False
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'Exam_Score', 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocessing pipeline for selected features only
        """
        print("Starting data preprocessing (selected features)...")
        # Select only the required columns
        required_cols = [
            'Hours_Studied', 'Previous_Scores', 'Attendance',
            'Extracurricular_Activities', 'Parental_Education_Level', target_column
        ]
        df = df[required_cols].copy()
        # Map Extracurricular_Activities to 1/0
        df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        # Map Parental_Education_Level
        edu_map = {'High School': 1, 'Undergraduate': 2, 'Graduate': 3, 1: 1, 2: 2, 3: 3}
        df['Parental_Education_Level'] = df['Parental_Education_Level'].map(edu_map).fillna(1).astype(int)
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        # Split data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.feature_columns = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.is_fitted = True
        print("✅ Data preprocessing (selected features) completed!")
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using only the selected features
        """
        required_cols = [
            'Hours_Studied', 'Previous_Scores', 'Attendance',
            'Extracurricular_Activities', 'Parental_Education_Level'
        ]
        df = df[required_cols].copy()
        df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0}).fillna(0).astype(int)
        edu_map = {'High School': 1, 'Undergraduate': 2, 'Graduate': 3, 1: 1, 2: 2, 3: 3}
        df['Parental_Education_Level'] = df['Parental_Education_Level'].map(edu_map).fillna(1).astype(int)
        # Fill missing
        df = df.fillna(df.mean(numeric_only=True))
        return df[self.feature_columns].values
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features used in the model"""
        return self.feature_columns.copy()
    
    def get_preprocessing_info(self) -> Dict[str, Any]:
        """Get information about the preprocessing steps"""
        return {
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'numeric_columns': self.numeric_columns,
            'label_encoders': list(self.label_encoders.keys()),
            'has_onehot_encoder': self.onehot_encoder is not None
        }

def main():
    """Test the data preprocessing module"""
    from data_collection import DataCollector
    
    print("=" * 60)
    print("DATA PREPROCESSING MODULE TEST")
    print("=" * 60)
    
    # Load data
    collector = DataCollector()
    df = collector.load_uci_dataset()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df)
    
    # Get preprocessing information
    info = preprocessor.get_preprocessing_info()
    
    print(f"\nPreprocessing Information:")
    print(f"Fitted: {info['is_fitted']}")
    print(f"Total features: {len(info['feature_columns'])}")
    print(f"Categorical features: {len(info['categorical_columns'])}")
    print(f"Numeric features: {len(info['numeric_columns'])}")
    print(f"Label encoded features: {len(info['label_encoders'])}")
    print(f"One-hot encoded: {info['has_onehot_encoder']}")
    
    print(f"\nFeature names (first 10):")
    for i, feature in enumerate(info['feature_columns'][:10]):
        print(f"  {i+1}. {feature}")
    
    print(f"\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    
    print(f"\n✅ Data preprocessing completed successfully!")

if __name__ == "__main__":
    main() 