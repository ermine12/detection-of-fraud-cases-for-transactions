"""
Explainability Script for Fraud_Data.csv Models

This script loads the best trained model from Task 2 and performs
comprehensive explainability analysis using SHAP.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from data_preprocessing import (
    load_data,
    clean_data,
    merge_geolocation,
    feature_engineering,
    transform_data
)
from model_explainability import full_explainability_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_best_model(model_dir: str = 'models/fraud_data'):
    """
    Load the best trained model and check which one exists.
    
    Returns:
        model, model_name
    """
    logger.info(f"Loading best model from {model_dir}...")
    
    # Check which models exist
    models_to_try = ['random_forest.pkl', 'xgboost.pkl', 'logistic_regression.pkl']
    
    for model_file in models_to_try:
        model_path = os.path.join(model_dir, model_file)
        if os.path.exists(model_path):
            logger.info(f"Loading {model_file}...")
            model = joblib.load(model_path)
            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
            logger.info(f"Successfully loaded: {model_name}")
            return model, model_name
    
    raise FileNotFoundError(
        f"No trained models found in {model_dir}. "
        f"Please run train_fraud_data.py first."
    )


def prepare_data_for_explainability():
    """
    Prepare data using the same preprocessing as training.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info("="*80)
    logger.info("PREPARING DATA FOR EXPLAINABILITY")
    logger.info("="*80)
    
    # Load and preprocess data (same as training)
    fraud_data, ip_country = load_data(
        fraud_path='data/raw/Fraud_Data.csv',
        ip_country_path='data/raw/IpAddress_to_Country.csv'
    )
    
    fraud_data, ip_country = clean_data(fraud_data, ip_country)
    fraud_data = merge_geolocation(fraud_data, ip_country)
    fraud_data = feature_engineering(fraud_data)
    
    # Drop unnecessary columns
    cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
    cols_present = [c for c in cols_to_drop if c in fraud_data.columns]
    fraud_data = fraud_data.drop(cols_present, axis=1)
    
    # Split features and target
    X = fraud_data.drop('class', axis=1)
    y = fraud_data['class']
    
    # Transform features
    X_transformed, scaler, encoder = transform_data(X.copy())
    
    # Convert to DataFrame for SHAP
    feature_names = X_transformed.columns.tolist()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Perform the same split as training (stratified, 80/20, same random_state)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names


def main():
    """Main explainability pipeline for Fraud_Data models."""
    logger.info("\n" + "="*80)
    logger.info("FRAUD DETECTION MODEL EXPLAINABILITY - FRAUD_DATA.CSV")
    logger.info("="*80 + "\n")
    
    try:
        # Step 1: Load best model
        model, model_name = load_best_model('models/fraud_data')
        
        # Step 2: Prepare data
        X_train, X_test, y_train, y_test, feature_names = prepare_data_for_explainability()
        
        # Step 3: Make predictions on test set
        logger.info("Making predictions on test set...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Step 4: Run explainability analysis
        results = full_explainability_pipeline(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            feature_names=feature_names,
            output_dir='reports',
            dataset_name='fraud_data'
        )
        
        logger.info("\n" + "="*80)
        logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Model analyzed: {model_name}")
        logger.info(f"Reports saved to: reports/fraud_data/")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
