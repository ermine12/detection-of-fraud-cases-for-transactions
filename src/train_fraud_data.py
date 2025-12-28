"""
Training script for Fraud_Data.csv

This script preprocesses the Fraud_Data.csv and trains multiple fraud detection models.
It demonstrates the complete pipeline from data loading to model selection.
"""

import os
import sys
import pandas as pd
import logging
from data_preprocessing import (
    load_data,
    clean_data,
    merge_geolocation,
    feature_engineering,
    transform_data
)
from model_training import full_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def preprocess_fraud_data():
    """
    Preprocess Fraud_Data.csv without applying SMOTE.
    SMOTE will be applied in the training pipeline to avoid data leakage.
    
    Returns:
        X, y: Feature matrix and target labels
    """
    logger.info("="*80)
    logger.info("PREPROCESSING FRAUD_DATA.CSV")
    logger.info("="*80)
    
    try:
        # Load data
        fraud_data, ip_country = load_data(
            fraud_path='data/raw/Fraud_Data.csv',
            ip_country_path='data/raw/IpAddress_to_Country.csv'
        )
        
        # Clean data
        fraud_data, ip_country = clean_data(fraud_data, ip_country)
        
        # Merge geolocation
        fraud_data = merge_geolocation(fraud_data, ip_country)
        
        # Feature engineering
        fraud_data = feature_engineering(fraud_data)
        
        # Drop unnecessary columns
        cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
        cols_present = [c for c in cols_to_drop if c in fraud_data.columns]
        fraud_data = fraud_data.drop(cols_present, axis=1)
        
        # Split features and target
        if 'class' not in fraud_data.columns:
            raise ValueError("Target column 'class' not found after preprocessing.")
        
        X = fraud_data.drop('class', axis=1)
        y = fraud_data['class']
        
        # Transform features
        X_transformed, scaler, encoder = transform_data(X.copy())
        
        logger.info(f"Preprocessing complete. Shape: {X_transformed.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X_transformed, y
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise


def main():
    """Main training pipeline for Fraud_Data.csv"""
    logger.info("\n" + "="*80)
    logger.info("FRAUD DETECTION MODEL TRAINING - FRAUD_DATA.CSV")
    logger.info("="*80 + "\n")
    
    try:
        # Step 1: Preprocess data
        X, y = preprocess_fraud_data()
        
        # Step 2: Train models with full pipeline
        results = full_training_pipeline(
            X=X,
            y=y,
            apply_smote_flag=True,  # Apply SMOTE in training pipeline (after split)
            use_xgboost=True,       # Train XGBoost if available
            save_models=True,       # Save trained models
            output_dir='models/fraud_data'  # Output directory for models
        )
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Best model: {results['best_model_name']}")
        logger.info(f"Models saved to: models/fraud_data/")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
