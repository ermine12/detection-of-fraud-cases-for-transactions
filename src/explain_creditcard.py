"""
Explainability Script for creditcard.csv Models

This script loads the best trained model from Task 2 and performs
comprehensive explainability analysis using SHAP.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model_explainability import full_explainability_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_best_model(model_dir: str = 'models/creditcard'):
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
        f"Please run train_creditcard.py first."
    )


def prepare_data_for_explainability():
    """
    Prepare creditcard.csv data using the same preprocessing as training.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info("="*80)
    logger.info("PREPARING DATA FOR EXPLAINABILITY")
    logger.info("="*80)
    
    # Load data
    data_path = 'data/raw/creditcard.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Credit card data not found: {data_path}")
    
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Scale Time and Amount features
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    feature_names = X.columns.tolist()
    
    # Perform the same split as training (stratified, 80/20, same random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    logger.info(f"Data prepared: {len(X_train)} train, {len(X_test)} test samples")
    logger.info(f"Features: {len(feature_names)}")
    
    return X_train, X_test, y_train, y_test, feature_names


def main():
    """Main explainability pipeline for creditcard models."""
    logger.info("\n" + "="*80)
    logger.info("FRAUD DETECTION MODEL EXPLAINABILITY - CREDITCARD.CSV")
    logger.info("="*80 + "\n")
    
    try:
        # Step 1: Load best model
        model, model_name = load_best_model('models/creditcard')
        
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
            dataset_name='creditcard'
        )
        
        logger.info("\n" + "="*80)
        logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Model analyzed: {model_name}")
        logger.info(f"Reports saved to: reports/creditcard/")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Explainability analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
