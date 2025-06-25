#!/usr/bin/env python3
"""
Data Collection Module for Student Performance Prediction

This module handles:
1. UCI Student Performance Dataset loading
2. Custom CSV data loading
3. Data validation and basic cleaning
4. Data source management
"""

import pandas as pd
import numpy as np
import requests
import os
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    """Handles data collection from various sources"""
    
    def __init__(self):
        self.data_sources = {
            'uci': {
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip',
                'description': 'UCI Student Performance Dataset'
            }
        }
    
    def load_uci_dataset(self, filepath: Optional[str] = None, n_samples: int = 1000) -> pd.DataFrame:
        """
        Load UCI Student Performance Dataset
        
        Args:
            filepath: Path to local CSV file (if already downloaded)
            n_samples: Number of samples to generate (for synthetic data)
            
        Returns:
            DataFrame with student performance data
        """
        if filepath and os.path.exists(filepath):
            print(f"Loading UCI dataset from local file: {filepath}")
            df = pd.read_csv(filepath, sep=';')
        else:
            print("UCI dataset not found locally. Using synthetic data with UCI structure...")
            df = self._generate_uci_style_data(n_samples)
        
        return self._clean_uci_data(df)
    
    def _generate_uci_style_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic data that mimics UCI Student Performance Dataset structure
        """
        np.random.seed(42)
        
        data = {
            # Demographics
            'school': np.random.choice(['GP', 'MS'], n_samples),
            'sex': np.random.choice(['F', 'M'], n_samples),
            'age': np.random.randint(15, 23, n_samples),
            'address': np.random.choice(['U', 'R'], n_samples),
            'famsize': np.random.choice(['LE3', 'GT3'], n_samples),
            'Pstatus': np.random.choice(['T', 'A'], n_samples),
            
            # Parental education and occupation
            'Medu': np.random.randint(0, 5, n_samples),  # Mother's education
            'Fedu': np.random.randint(0, 5, n_samples),  # Father's education
            'Mjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'Fjob': np.random.choice(['teacher', 'health', 'services', 'at_home', 'other'], n_samples),
            'reason': np.random.choice(['home', 'reputation', 'course', 'other'], n_samples),
            'guardian': np.random.choice(['mother', 'father', 'other'], n_samples),
            
            # Academic factors
            'traveltime': np.random.randint(1, 5, n_samples),
            'studytime': np.random.randint(1, 5, n_samples),
            'failures': np.random.randint(0, 4, n_samples),
            'schoolsup': np.random.choice(['yes', 'no'], n_samples),
            'famsup': np.random.choice(['yes', 'no'], n_samples),
            'paid': np.random.choice(['yes', 'no'], n_samples),
            'activities': np.random.choice(['yes', 'no'], n_samples),
            'nursery': np.random.choice(['yes', 'no'], n_samples),
            'higher': np.random.choice(['yes', 'no'], n_samples),
            'internet': np.random.choice(['yes', 'no'], n_samples),
            'romantic': np.random.choice(['yes', 'no'], n_samples),
            
            # Health and lifestyle
            'famrel': np.random.randint(1, 6, n_samples),
            'freetime': np.random.randint(1, 6, n_samples),
            'goout': np.random.randint(1, 6, n_samples),
            'Dalc': np.random.randint(1, 6, n_samples),  # Workday alcohol consumption
            'Walc': np.random.randint(1, 6, n_samples),  # Weekend alcohol consumption
            'health': np.random.randint(1, 6, n_samples),
            'absences': np.random.randint(0, 30, n_samples),
            
            # Past grades (strong predictors)
            'G1': np.random.normal(12, 3, n_samples).clip(0, 20),
            'G2': np.random.normal(12, 3, n_samples).clip(0, 20),
            
            # Target variable
            'G3': np.random.normal(12, 3, n_samples).clip(0, 20)
        }
        
        # Create realistic relationships
        df = pd.DataFrame(data)
        
        # G3 (final grade) should be related to G1 and G2
        base_performance = (df['G1'] + df['G2']) / 2
        study_effect = df['studytime'] * 0.5
        attendance_effect = (20 - df['absences']) * 0.1
        health_effect = (df['health'] - 3) * 0.3
        internet_effect = (df['internet'] == 'yes').astype(int) * 0.5
        
        # Combine effects with some randomness
        df['G3'] = (base_performance + study_effect + attendance_effect + 
                   health_effect + internet_effect + np.random.normal(0, 1, n_samples)).clip(0, 20)
        
        return df
    
    def _clean_uci_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate UCI dataset
        """
        print(f"Original dataset shape: {df.shape}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found:\n{missing_values[missing_values > 0]}")
            # Fill missing values
            df = df.fillna(df.mode().iloc[0])
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                          'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                          'absences', 'G1', 'G2', 'G3']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid grades
        df = df[df['G3'].between(0, 20)]
        
        print(f"Cleaned dataset shape: {df.shape}")
        return df
    
    def load_custom_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load custom CSV file with student data
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with student data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Loading custom dataset from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Basic validation
        required_columns = ['G3']  # At minimum, we need the target variable
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")
        
        return df
    
    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'target_column': 'G3' if 'G3' in df.columns else None
        }
        
        if info['target_column']:
            target_stats = df[info['target_column']].describe()
            info['target_statistics'] = target_stats.to_dict()
        
        return info
    
    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save dataset to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path where to save the file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")

def main():
    """Test the data collection module"""
    collector = DataCollector()
    
    # Load UCI-style dataset
    print("=" * 60)
    print("DATA COLLECTION MODULE TEST")
    print("=" * 60)
    
    df = collector.load_uci_dataset(n_samples=100)
    
    # Get dataset information
    info = collector.get_data_info(df)
    
    print(f"\nDataset Information:")
    print(f"Shape: {info['shape']}")
    print(f"Target column: {info['target_column']}")
    print(f"Numeric columns: {len(info['numeric_columns'])}")
    print(f"Categorical columns: {len(info['categorical_columns'])}")
    
    if info['target_column']:
        print(f"\nTarget variable statistics:")
        for stat, value in info['target_statistics'].items():
            print(f"  {stat}: {value:.2f}")
    
    # Save dataset
    collector.save_dataset(df, 'student_performance_data.csv')
    
    print(f"\n✅ Data collection completed successfully!")
    print(f"Dataset saved as: student_performance_data.csv")

if __name__ == "__main__":
    main() 