"""
Model Training Module for Fraud Detection

This module provides comprehensive functionality for training, evaluating,
and comparing fraud detection models including baseline and ensemble methods.

Features:
- Stratified train-test splits
- Baseline Logistic Regression
- Ensemble models (Random Forest, XGBoost)
- Hyperparameter tuning with GridSearchCV
- Stratified K-Fold cross-validation
- Comprehensive metrics (ROC-AUC, AUC-PR, F1, Precision, Recall)
- Model comparison and selection
"""

import os
import logging
from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    make_scorer
)
from imblearn.over_sampling import SMOTE
import warnings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train-test split to preserve class distribution.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data: {len(X)} samples, test_size={test_size}")
    logger.info(f"Original class distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Train class distribution:\n{y_train.value_counts()}")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Test class distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to training data only (avoid data leakage).
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed
        
    Returns:
        X_train_resampled, y_train_resampled
    """
    logger.info("Applying SMOTE to training data...")
    logger.info(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"After SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    
    return pd.DataFrame(X_train_resampled, columns=X_train.columns), pd.Series(y_train_resampled)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_proba),
        'auc_pr': average_precision_score(y_true, y_proba),
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    return metrics


def print_evaluation_report(
    model_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Print comprehensive evaluation report for a model.
    
    Args:
        model_name: Name of the model
        y_test: True test labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Report: {model_name}")
    logger.info(f"{'='*60}")
    
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    logger.info(f"ROC-AUC Score:    {metrics['roc_auc']:.4f}")
    logger.info(f"AUC-PR Score:     {metrics['auc_pr']:.4f}")
    logger.info(f"F1 Score:         {metrics['f1_score']:.4f}")
    logger.info(f"Precision:        {metrics['precision']:.4f}")
    logger.info(f"Recall:           {metrics['recall']:.4f}")
    
    logger.info(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")
    
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return metrics


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Any, Dict[str, float]]:
    """
    Train Logistic Regression baseline model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Trained model and metrics dictionary
    """
    logger.info("\n" + "="*60)
    logger.info("Training Logistic Regression (Baseline)")
    logger.info("="*60)
    
    # Use class_weight='balanced' to handle any remaining imbalance
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    
    logger.info("Fitting model...")
    model.fit(X_train, y_train)
    
    logger.info("Making predictions on test set...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = print_evaluation_report("Logistic Regression", y_test, y_pred, y_proba)
    
    return model, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict[str, List] = None
) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
    """
    Train Random Forest with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        
    Returns:
        Best model, metrics dictionary, and best parameters
    """
    logger.info("\n" + "="*60)
    logger.info("Training Random Forest with Hyperparameter Tuning")
    logger.info("="*60)
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
    
    logger.info(f"Parameter grid: {param_grid}")
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    logger.info("Running GridSearchCV with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    logger.info("Making predictions on test set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = print_evaluation_report("Random Forest", y_test, y_pred, y_proba)
    
    return best_model, metrics, grid_search.best_params_


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_grid: Dict[str, List] = None
) -> Tuple[Any, Dict[str, float], Dict[str, Any]]:
    """
    Train XGBoost with hyperparameter tuning.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        
    Returns:
        Best model, metrics dictionary, and best parameters
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    logger.info("\n" + "="*60)
    logger.info("Training XGBoost with Hyperparameter Tuning")
    logger.info("="*60)
    
    if param_grid is None:
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'scale_pos_weight': [scale_pos_weight]
        }
    
    logger.info(f"Parameter grid: {param_grid}")
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    logger.info("Running GridSearchCV with 5-fold cross-validation...")
    grid_search = GridSearchCV(
        xgb,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    logger.info("Making predictions on test set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = print_evaluation_report("XGBoost", y_test, y_pred, y_proba)
    
    return best_model, metrics, grid_search.best_params_


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, np.ndarray]:
    """
    Perform stratified k-fold cross-validation and report mean ± std.
    
    Args:
        model: Scikit-learn compatible model
        X: Features
        y: Labels
        cv: Number of folds
        
    Returns:
        Dictionary of cross-validation scores
    """
    logger.info(f"\nPerforming {cv}-fold stratified cross-validation...")
    
    # Define scoring metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }
    
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = cross_validate(
        model,
        X, y,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    
    # Print results
    logger.info(f"\nCross-Validation Results ({cv} folds):")
    logger.info("-" * 60)
    
    for metric_name, scores in cv_results.items():
        if metric_name.startswith('test_'):
            clean_name = metric_name.replace('test_', '')
            mean_score = scores.mean()
            std_score = scores.std()
            logger.info(f"{clean_name:20s}: {mean_score:.4f} ± {std_score:.4f}")
    
    return cv_results


def compare_models(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of model performances.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        DataFrame with comparison
    """
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)
    
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.round(4)
    
    # Sort by ROC-AUC (primary metric)
    comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
    
    logger.info(f"\n{comparison_df.to_string()}")
    
    return comparison_df


def select_best_model(
    comparison_df: pd.DataFrame,
    models: Dict[str, Any]
) -> Tuple[str, Any]:
    """
    Select the best model with justification.
    
    Args:
        comparison_df: Model comparison DataFrame
        models: Dictionary of trained models
        
    Returns:
        Name of best model and the model object
    """
    logger.info("\n" + "="*60)
    logger.info("Model Selection")
    logger.info("="*60)
    
    # Primary ranking by ROC-AUC
    best_model_name = comparison_df.index[0]
    best_roc_auc = comparison_df.loc[best_model_name, 'roc_auc']
    
    logger.info(f"\nSelected Model: {best_model_name}")
    logger.info(f"\nJustification:")
    logger.info(f"- Highest ROC-AUC score: {best_roc_auc:.4f}")
    logger.info(f"- AUC-PR: {comparison_df.loc[best_model_name, 'auc_pr']:.4f}")
    logger.info(f"- F1 Score: {comparison_df.loc[best_model_name, 'f1_score']:.4f}")
    
    # Check if Logistic Regression is competitive (within 2% of best)
    if 'Logistic Regression' in comparison_df.index:
        lr_roc_auc = comparison_df.loc['Logistic Regression', 'roc_auc']
        if best_model_name != 'Logistic Regression' and (best_roc_auc - lr_roc_auc) < 0.02:
            logger.info(f"\nNote: Logistic Regression (ROC-AUC={lr_roc_auc:.4f}) is competitive")
            logger.info(f"and offers better interpretability. Consider using it for production.")
    
    return best_model_name, models[best_model_name]


def save_model(model: Any, model_name: str, output_dir: str = 'models') -> str:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Directory to save model
        
    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean model name for filename
    filename = model_name.lower().replace(' ', '_') + '.pkl'
    filepath = os.path.join(output_dir, filename)
    
    logger.info(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    logger.info("Model saved successfully!")
    
    return filepath


def full_training_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    apply_smote_flag: bool = True,
    use_xgboost: bool = True,
    save_models: bool = True,
    output_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Complete training pipeline with all models.
    
    Args:
        X: Feature matrix
        y: Target labels
        apply_smote_flag: Whether to apply SMOTE to training data
        use_xgboost: Whether to train XGBoost (requires xgboost package)
        save_models: Whether to save trained models
        output_dir: Directory for saving models
        
    Returns:
        Dictionary containing models, metrics, and comparison results
    """
    # Step 1: Stratified train-test split
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y)
    
    # Step 2: Apply SMOTE to training data only (if requested)
    if apply_smote_flag:
        X_train, y_train = apply_smote(X_train, y_train)
    
    # Step 3: Train baseline model
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    # Step 4: Train ensemble models
    rf_model, rf_metrics, rf_params = train_random_forest(X_train, y_train, X_test, y_test)
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {
        'Logistic Regression': lr_metrics,
        'Random Forest': rf_metrics
    }
    
    if use_xgboost and XGBOOST_AVAILABLE:
        xgb_model, xgb_metrics, xgb_params = train_xgboost(X_train, y_train, X_test, y_test)
        models['XGBoost'] = xgb_model
        results['XGBoost'] = xgb_metrics
    
    # Step 5: Cross-validation on best performing model (based on test set)
    comparison_df = compare_models(results)
    best_name_preliminary = comparison_df.index[0]
    logger.info(f"\nPerforming cross-validation on {best_name_preliminary}...")
    cross_validate_model(models[best_name_preliminary], X, y, cv=5)
    
    # Step 6: Model selection
    best_model_name, best_model = select_best_model(comparison_df, models)
    
    # Step 7: Save models
    if save_models:
        for name, model in models.items():
            save_model(model, name, output_dir)
    
    return {
        'models': models,
        'metrics': results,
        'comparison': comparison_df,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'test_data': (X_test, y_test)
    }


if __name__ == '__main__':
    # Example usage
    logger.info("This module is meant to be imported. See train_fraud_data.py or train_creditcard.py for usage examples.")
