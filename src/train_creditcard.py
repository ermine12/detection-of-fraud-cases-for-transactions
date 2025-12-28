"""
Training script for creditcard.csv

This script preprocesses the creditcard.csv and trains multiple fraud detection models.
The creditcard dataset is already numeric and anonymized, requiring minimal preprocessing.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from model_training import full_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_creditcard_data():
    """
    Preprocess creditcard.csv.
    The dataset is already numeric (PCA-transformed features V1-V28, Time, Amount, Class).
    We only need to scale Time and Amount features.
    
    Returns:
        X, y: Feature matrix and target labels
    """
    logger.info("="*80)
    logger.info("PREPROCESSING CREDITCARD.CSV")
    logger.info("="*80)
    
    try:
        # Load data
        data_path = 'data/raw/creditcard.csv'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Credit card data not found: {data_path}")
        
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} transactions")
        
        # Check for required columns
        if 'Class' not in df.columns:
            raise ValueError("Target column 'Class' not found in creditcard.csv")
        
        # Report class distribution
        class_dist = df['Class'].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")
        logger.info(f"Fraud rate: {(class_dist[1] / len(df) * 100):.2f}%")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Scale Time and Amount features (V1-V28 are already scaled by PCA)
        logger.info("Scaling 'Time' and 'Amount' features...")
        scaler = StandardScaler()
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
        
        logger.info(f"Preprocessing complete. Shape: {X.shape}")
        
        return X, y
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def main():
    """Main training pipeline for creditcard.csv"""
    logger.info("\n" + "="*80)
    logger.info("FRAUD DETECTION MODEL TRAINING - CREDITCARD.CSV")
    logger.info("="*80 + "\n")
    
    try:
        # Step 1: Preprocess data
        X, y = preprocess_creditcard_data()
        
        # Step 2: Train models with full pipeline
        # Note: Due to large size of creditcard.csv, we use fewer hyperparameter combinations
        results = full_training_pipeline(
            X=X,
            y=y,
            apply_smote_flag=True,  # Apply SMOTE in training pipeline (after split)
            use_xgboost=True,       # Train XGBoost if available
            save_models=True,       # Save trained models
            output_dir='models/creditcard'  # Output directory for models
        )
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best model: {results['best_model_name']}")
        logger.info(f"Models saved to: models/creditcard/")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
