#!/usr/bin/env python3
"""
Main Training Script for Student Performance Prediction

This script orchestrates the complete pipeline:
1. Data Collection
2. Data Preprocessing
3. Model Training
4. Model Evaluation
5. Performance Analysis
6. Model Saving
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_collection import DataCollector
from data_preprocessing import DataPreprocessor
from neural_network_model import StudentPerformanceModel

class ModelTrainer:
    """Main class for training the student performance prediction model"""
    
    def __init__(self, config: dict = None):
        """
        Initialize the trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.collector = DataCollector()
        self.preprocessor = DataPreprocessor()
        self.model = None
        self.results = {}
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
    
    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'data': {
                'n_samples': 1000,
                'test_size': 0.2,
                'random_state': 42
            },
            'model': {
                'problem_type': 'regression',
                'architecture': {
                    'hidden_layers': [64, 32, 16],
                    'dropout_rate': 0.3,
                    'learning_rate': 0.001,
                    'activation': 'relu',
                    'output_activation': 'linear'
                }
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'validation_split': 0.2,
                'early_stopping_patience': 15,
                'lr_scheduler_patience': 10
            },
            'evaluation': {
                'save_plots': True,
                'save_model': True,
                'generate_report': True
            }
        }
    
    def run_pipeline(self) -> dict:
        """
        Run the complete training pipeline
        
        Returns:
            Dictionary with results
        """
        print("=" * 80)
        print("STUDENT PERFORMANCE PREDICTION - COMPLETE PIPELINE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Data Collection
            print("\n1. DATA COLLECTION")
            print("-" * 40)
            df = self._collect_data()
            
            # Step 2: Data Preprocessing
            print("\n2. DATA PREPROCESSING")
            print("-" * 40)
            X_train, X_test, y_train, y_test = self._preprocess_data(df)
            
            # Step 3: Model Training
            print("\n3. MODEL TRAINING")
            print("-" * 40)
            self._train_model(X_train, y_train)
            
            # Step 4: Model Evaluation
            print("\n4. MODEL EVALUATION")
            print("-" * 40)
            self._evaluate_model(X_test, y_test)
            
            # Step 5: Performance Analysis
            print("\n5. PERFORMANCE ANALYSIS")
            print("-" * 40)
            self._analyze_performance(X_test, y_test)
            
            # Step 6: Save Results
            print("\n6. SAVING RESULTS")
            print("-" * 40)
            self._save_results()
            
            print(f"\n✅ Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return self.results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise
    
    def _collect_data(self) -> pd.DataFrame:
        """Collect and prepare the dataset"""
        print("Loading UCI-style student performance dataset...")
        df = self.collector.load_uci_dataset()
        
        # Get dataset information
        info = self.collector.get_data_info(df)
        self.results['data_info'] = info
        
        print(f"Dataset loaded: {info['shape'][0]} samples, {info['shape'][1]} features")
        print(f"Target variable: {info['target_column']}")
        
        # Save raw data
        self.collector.save_dataset(df, 'data/raw_student_data.csv')
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess the dataset"""
        print("Preprocessing data...")
        
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(
            df,
            target_column='G3',
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        
        # Store preprocessing information
        self.results['preprocessing_info'] = self.preprocessor.get_preprocessing_info()
        self.results['feature_names'] = self.preprocessor.get_feature_names()
        
        print(f"Data preprocessed: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
        print(f"Features: {X_train.shape[1]} columns")
        
        return X_train, X_test, y_train, y_test
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the neural network model"""
        print("Initializing neural network model...")
        
        # Initialize model
        self.model = StudentPerformanceModel(
            input_dim=X_train.shape[1],
            problem_type=self.config['model']['problem_type']
        )
        
        # Build model
        self.model.build_model(self.config['model']['architecture'])
        
        # Print model summary
        print("\nModel Architecture:")
        print(self.model.get_model_summary())
        
        # Train model
        print("\nTraining model...")
        history = self.model.train(X_train, y_train, training_config=self.config['training'])
        
        # Store training history
        self.results['training_history'] = history.history
        self.results['model_architecture'] = self.config['model']['architecture']
        
        print("Model training completed!")
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Evaluate the trained model"""
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = self.model.evaluate(X_test, y_test)
        self.results['test_metrics'] = metrics
        self.results['predictions'] = {
            'y_true': y_test,
            'y_pred': y_pred.flatten()
        }
        
        # Print results
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # Get feature importance
        feature_names = self.results['feature_names']
        importance = self.model.get_feature_importance(feature_names)
        self.results['feature_importance'] = importance
        
        print(f"\nTop 10 Most Important Features:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_importance[:10], 1):
            print(f"  {i}. {feature}: {score:.4f}")
    
    def _analyze_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Analyze model performance and create visualizations"""
        print("Creating performance visualizations...")
        
        y_pred = self.model.predict(X_test)
        
        # Plot training history
        if self.config['evaluation']['save_plots']:
            self.model.plot_training_history(save_path='plots/training_history.png')
            self.model.plot_predictions(y_test, y_pred, save_path='plots/predictions.png')
            
            # Feature importance plot
            self._plot_feature_importance()
            
            # Correlation analysis
            self._plot_correlation_analysis()
    
    def _plot_feature_importance(self) -> None:
        """Plot feature importance"""
        importance = self.results['feature_importance']
        
        # Get top 15 features
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
        features, scores = zip(*sorted_importance)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), scores, color='skyblue', edgecolor='navy')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Student Performance Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self) -> None:
        """Plot correlation analysis"""
        # Load original data for correlation analysis
        df = self.collector.load_uci_dataset()
        
        # Calculate correlations with target
        correlations = df.corr()['G3'].sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        correlations[correlations.index != 'G3'].plot(kind='bar', color='lightcoral')
        plt.title('Feature Correlations with Final Grade (G3)')
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('plots/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self) -> None:
        """Save all results and artifacts"""
        print("Saving results and artifacts...")
        
        # Save model
        if self.config['evaluation']['save_model']:
            model_path = f"models/student_performance_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            self.model.save_model(model_path)
            self.results['model_path'] = model_path
        
        # Save preprocessor
        preprocessor_path = f"models/preprocessor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        import joblib
        joblib.dump(self.preprocessor, preprocessor_path)
        self.results['preprocessor_path'] = preprocessor_path
        
        # Save results summary
        self._save_results_summary()
        
        # Generate report
        if self.config['evaluation']['generate_report']:
            self._generate_report()
    
    def _save_results_summary(self) -> None:
        """Save a summary of results"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'data_info': self.results['data_info'],
            'preprocessing_info': self.results['preprocessing_info'],
            'test_metrics': self.results['test_metrics'],
            'top_features': dict(sorted(self.results['feature_importance'].items(), 
                                      key=lambda x: x[1], reverse=True)[:10])
        }
        
        import json
        with open('reports/results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def _generate_report(self) -> None:
        """Generate a comprehensive report"""
        report = f"""
# Student Performance Prediction - Model Report

## Executive Summary
- **Model Type**: Neural Network ({self.config['model']['problem_type']})
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dataset Size**: {self.results['data_info']['shape'][0]} samples
- **Features**: {len(self.results['feature_names'])} engineered features

## Model Performance
"""
        
        for metric, value in self.results['test_metrics'].items():
            report += f"- **{metric.upper()}**: {value:.4f}\n"
        
        report += f"""
## Model Architecture
- **Hidden Layers**: {self.config['model']['architecture']['hidden_layers']}
- **Dropout Rate**: {self.config['model']['architecture']['dropout_rate']}
- **Learning Rate**: {self.config['model']['architecture']['learning_rate']}
- **Activation**: {self.config['model']['architecture']['activation']}

## Top 10 Most Important Features
"""
        
        sorted_importance = sorted(self.results['feature_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, score) in enumerate(sorted_importance, 1):
            report += f"{i}. **{feature}**: {score:.4f}\n"
        
        report += f"""
## Files Generated
- Model: {self.results.get('model_path', 'N/A')}
- Preprocessor: {self.results.get('preprocessor_path', 'N/A')}
- Plots: plots/
- Reports: reports/

## Next Steps
1. Deploy the model using the saved files
2. Monitor model performance in production
3. Retrain periodically with new data
4. Consider ensemble methods for improved performance
"""
        
        with open('reports/model_report.md', 'w') as f:
            f.write(report)

def main():
    """Main function to run the training pipeline"""
    # Configuration
    config = {
        'data': {
            'n_samples': 1000,
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'problem_type': 'regression',
            'architecture': {
                'hidden_layers': [64, 32, 16],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'activation': 'relu',
                'output_activation': 'linear'
            }
        },
        'training': {
            'epochs': 50,  # Reduced for faster training
            'batch_size': 32,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'lr_scheduler_patience': 5
        },
        'evaluation': {
            'save_plots': True,
            'save_model': True,
            'generate_report': True
        }
    }
    
    # Create trainer and run pipeline
    trainer = ModelTrainer(config)
    results = trainer.run_pipeline()
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("Next steps:")
    print("1. Check the 'models/' directory for saved model files")
    print("2. Check the 'plots/' directory for visualizations")
    print("3. Check the 'reports/' directory for detailed reports")
    print("4. Run 'streamlit run app.py' to start the web interface")

if __name__ == "__main__":
    main() 