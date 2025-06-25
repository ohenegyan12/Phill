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
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = 'G3', 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test: Preprocessed and split data
        """
        print("Starting data preprocessing...")
        
        # Step 1: Clean data
        df_clean = self._clean_data(df)
        
        # Step 2: Feature engineering
        df_engineered = self._engineer_features(df_clean)
        
        # Step 3: Encode categorical variables
        df_encoded = self._encode_categorical_variables(df_engineered)
        
        # Step 4: Handle missing values
        df_imputed = self._handle_missing_values(df_encoded)
        
        # Step 5: Scale features
        df_scaled = self._scale_features(df_imputed)
        
        # Step 6: Split data
        X_train, X_test, y_train, y_test = self._split_data(df_scaled, target_column, test_size, random_state)
        
        self.is_fitted = True
        print("✅ Data preprocessing completed successfully!")
        
        return X_train, X_test, y_train, y_test
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        print("  Cleaning data...")
        
        # Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        if df.shape != initial_shape:
            print(f"    Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        
        # Handle outliers in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        print("  Engineering features...")
        
        # Academic performance indicators
        if 'G1' in df.columns and 'G2' in df.columns:
            df['avg_previous_grades'] = (df['G1'] + df['G2']) / 2
            df['grade_improvement'] = df['G2'] - df['G1']
        
        # Study efficiency (study time vs absences)
        if 'studytime' in df.columns and 'absences' in df.columns:
            df['study_efficiency'] = df['studytime'] / (df['absences'] + 1)
        
        # Family support composite
        if 'famsup' in df.columns and 'famrel' in df.columns:
            df['family_support'] = ((df['famsup'] == 'yes').astype(int) + df['famrel']) / 2
        
        # Lifestyle factors
        if 'Dalc' in df.columns and 'Walc' in df.columns:
            df['alcohol_consumption'] = (df['Dalc'] + df['Walc']) / 2
        
        # Social activity level
        if 'goout' in df.columns and 'freetime' in df.columns:
            df['social_activity'] = (df['goout'] + df['freetime']) / 2
        
        # Risk factors (failures, absences, low health)
        risk_factors = []
        if 'failures' in df.columns:
            risk_factors.append(df['failures'])
        if 'absences' in df.columns:
            risk_factors.append(df['absences'] / 30)  # Normalize absences
        if 'health' in df.columns:
            risk_factors.append((6 - df['health']) / 5)  # Invert health (lower = higher risk)
        
        if risk_factors:
            df['risk_score'] = np.mean(risk_factors, axis=0)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables using appropriate methods"""
        print("  Encoding categorical variables...")
        
        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.categorical_columns = categorical_cols
        
        # Binary variables (yes/no)
        binary_cols = []
        for col in categorical_cols:
            if df[col].nunique() == 2:
                binary_cols.append(col)
        
        # Multi-category variables
        multi_cat_cols = [col for col in categorical_cols if col not in binary_cols]
        
        # Encode binary variables
        for col in binary_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
        
        # Encode multi-category variables using one-hot encoding
        if multi_cat_cols:
            self.onehot_encoder = OneHotEncoder(drop='first', sparse=False)
            encoded_features = self.onehot_encoder.fit_transform(df[multi_cat_cols])
            
            # Create feature names
            feature_names = []
            for i, col in enumerate(multi_cat_cols):
                categories = self.onehot_encoder.categories_[i][1:]  # Drop first category
                feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Add encoded features to dataframe
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
            df = pd.concat([df.drop(columns=multi_cat_cols), encoded_df], axis=1)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        print("  Handling missing values...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"    Found missing values:\n{missing_counts[missing_counts > 0]}")
            
            # For numeric columns, use mean imputation
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
            
            # For categorical columns, use mode imputation
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().sum() > 0:
                    mode_value = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_value)
        else:
            print("    No missing values found")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features"""
        print("  Scaling features...")
        
        # Identify numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'G3' in numeric_cols:
            numeric_cols.remove('G3')
        
        self.numeric_columns = numeric_cols
        
        if numeric_cols:
            # Scale numeric features
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        return df
    
    def _split_data(self, df: pd.DataFrame, target_column: str, 
                   test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets"""
        print("  Splitting data...")
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Store feature column names
        self.feature_columns = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"    Training set: {X_train.shape[0]} samples")
        print(f"    Testing set: {X_test.shape[0]} samples")
        print(f"    Features: {X_train.shape[1]} columns")
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessors
        
        Args:
            df: New data to transform
            
        Returns:
            Transformed features as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming new data")
        
        # Apply the same preprocessing steps
        df_clean = self._clean_data(df.copy())
        df_engineered = self._engineer_features(df_clean)
        df_encoded = self._encode_categorical_variables(df_engineered)
        df_imputed = self._handle_missing_values(df_encoded)
        df_scaled = self._scale_features(df_imputed)
        
        # Ensure all expected features are present
        missing_features = set(self.feature_columns) - set(df_scaled.columns)
        if missing_features:
            # Add missing features with default values
            for feature in missing_features:
                df_scaled[feature] = 0
        
        # Select only the expected features in the correct order
        df_scaled = df_scaled[self.feature_columns]
        
        return df_scaled.values
    
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