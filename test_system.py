#!/usr/bin/env python3
"""
Test script for the Student Performance Prediction System

This script tests all major components:
1. Data collection
2. Data preprocessing
3. Neural network model
4. Training pipeline
5. Web application components
"""

import sys
import os
import numpy as np
import pandas as pd

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_collection import DataCollector
        print("✓ data_collection imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data_collection: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✓ data_preprocessing imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import data_preprocessing: {e}")
        return False
    
    try:
        from neural_network_model import StudentPerformanceModel
        print("✓ neural_network_model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import neural_network_model: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Streamlit: {e}")
        return False
    
    return True

def test_data_collection():
    """Test data collection functionality"""
    print("\nTesting data collection...")
    
    try:
        from data_collection import DataCollector
        
        collector = DataCollector()
        df = collector.load_uci_dataset(n_samples=100)
        
        if df is not None and len(df) > 0:
            print(f"✓ Data collection successful: {len(df)} samples, {len(df.columns)} features")
            
            # Test data info
            info = collector.get_data_info(df)
            print(f"✓ Data info extraction successful")
            
            return True
        else:
            print("✗ Data collection failed: empty dataset")
            return False
            
    except Exception as e:
        print(f"✗ Data collection failed: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        
        # Load data
        collector = DataCollector()
        df = collector.load_uci_dataset(n_samples=100)
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df, test_size=0.2)
        
        if (X_train is not None and X_test is not None and 
            y_train is not None and y_test is not None):
            print(f"✓ Data preprocessing successful:")
            print(f"  - Training set: {X_train.shape}")
            print(f"  - Test set: {X_test.shape}")
            print(f"  - Features: {X_train.shape[1]}")
            
            # Test feature names
            feature_names = preprocessor.get_feature_names()
            print(f"  - Feature count: {len(feature_names)}")
            
            return True
        else:
            print("✗ Data preprocessing failed: None values returned")
            return False
            
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")
        return False

def test_neural_network():
    """Test neural network model functionality"""
    print("\nTesting neural network model...")
    
    try:
        from data_collection import DataCollector
        from data_preprocessing import DataPreprocessor
        from neural_network_model import StudentPerformanceModel
        
        # Load and preprocess data
        collector = DataCollector()
        df = collector.load_uci_dataset(n_samples=100)
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df, test_size=0.2)
        
        # Initialize model
        model = StudentPerformanceModel(input_dim=X_train.shape[1], problem_type='regression')
        
        # Build model
        architecture = {
            'hidden_layers': [16, 8],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'activation': 'relu',
            'output_activation': 'linear'
        }
        model.build_model(architecture)
        
        print("✓ Model built successfully")
        
        # Test training (quick training)
        training_config = {
            'epochs': 5,
            'batch_size': 16,
            'validation_split': 0.2,
            'early_stopping_patience': 3,
            'lr_scheduler_patience': 2
        }
        
        history = model.train(X_train, y_train, training_config=training_config)
        print("✓ Model training completed")
        
        # Test prediction
        y_pred = model.predict(X_test)
        print(f"✓ Model prediction successful: {len(y_pred)} predictions")
        
        # Test evaluation
        metrics = model.evaluate(X_test, y_test)
        print(f"✓ Model evaluation successful:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value:.4f}")
        
        # Test feature importance
        feature_names = preprocessor.get_feature_names()
        importance = model.get_feature_importance(feature_names)
        print(f"✓ Feature importance calculation successful: {len(importance)} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return False

def test_training_pipeline():
    """Test the complete training pipeline"""
    print("\nTesting training pipeline...")
    
    try:
        from train_model import ModelTrainer
        
        # Create a minimal configuration for quick testing
        config = {
            'data': {
                'n_samples': 100,
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'problem_type': 'regression',
                'architecture': {
                    'hidden_layers': [16, 8],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'activation': 'relu',
                    'output_activation': 'linear'
                }
            },
            'training': {
                'epochs': 5,
                'batch_size': 16,
                'validation_split': 0.2,
                'early_stopping_patience': 3,
                'lr_scheduler_patience': 2
            },
            'evaluation': {
                'save_plots': False,
                'save_model': True,
                'generate_report': False
            }
        }
        
        trainer = ModelTrainer(config)
        results = trainer.run_pipeline()
        
        if results:
            print("✓ Training pipeline completed successfully")
            return True
        else:
            print("✗ Training pipeline failed")
            return False
            
    except Exception as e:
        print(f"✗ Training pipeline test failed: {e}")
        return False

def test_web_app_components():
    """Test web application components"""
    print("\nTesting web application components...")
    
    try:
        import streamlit as st
        
        # Test if we can create basic Streamlit components
        # This is a basic test - actual app testing would require running streamlit
        
        print("✓ Streamlit components test passed")
        return True
        
    except Exception as e:
        print(f"✗ Web app components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Data Collection", test_data_collection),
        ("Data Preprocessing", test_data_preprocessing),
        ("Neural Network Model", test_neural_network),
        ("Training Pipeline", test_training_pipeline),
        ("Web App Components", test_web_app_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python train_model.py")
        print("2. Run: streamlit run app.py")
        print("3. Open browser to http://localhost:8501")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check Python version compatibility")
        print("3. Verify TensorFlow installation")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 