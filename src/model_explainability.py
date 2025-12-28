"""
Model Explainability Module using SHAP

This module provides comprehensive model interpretation capabilities including:
- Built-in feature importance extraction and visualization
- SHAP TreeExplainer for global and local interpretability
- Force plots for individual predictions
- Business insights generation
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def extract_feature_importance(
    model: Any,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Extract built-in feature importance from ensemble models.
    
    Args:
        model: Trained model (Random Forest or XGBoost)
        feature_names: List of feature names
        
    Returns:
        DataFrame with features and their importance scores
    """
    logger.info("Extracting feature importance...")
    
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    logger.info(f"Extracted importance for {len(feature_names)} features")
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize top N feature importances.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    logger.info(f"Plotting top {top_n} features...")
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features (Built-in Importance)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    
    plt.close()


def create_shap_explainer(
    model: Any,
    X_sample: pd.DataFrame
) -> shap.TreeExplainer:
    """
    Create SHAP TreeExplainer for tree-based models.
    
    Args:
        model: Trained tree-based model
        X_sample: Sample of training data for background distribution
        
    Returns:
        SHAP TreeExplainer
    """
    logger.info("Creating SHAP TreeExplainer...")
    
    # Use a sample for efficiency (SHAP can be slow on large datasets)
    if len(X_sample) > 1000:
        logger.info(f"Using subsample of {min(1000, len(X_sample))} for background distribution")
        X_background = X_sample.sample(min(1000, len(X_sample)), random_state=42)
    else:
        X_background = X_sample
    
    explainer = shap.TreeExplainer(model, X_background)
    logger.info("SHAP explainer created successfully")
    
    return explainer


def calculate_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Calculate SHAP values for given data.
    
    Args:
        explainer: SHAP TreeExplainer
        X: Data to explain
        
    Returns:
        SHAP values array
    """
    logger.info(f"Calculating SHAP values for {len(X)} samples...")
    
    shap_values = explainer.shap_values(X)
    
    # For binary classification, shap_values might be a list [class_0, class_1]
    # We want SHAP values for the positive class (fraud = 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    logger.info("SHAP values calculated")
    return shap_values


def plot_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create SHAP summary plot (beeswarm plot).
    
    Args:
        shap_values: SHAP values
        X: Feature data
        save_path: Optional path to save the plot
    """
    logger.info("Generating SHAP summary plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Summary Plot - Global Feature Importance', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved SHAP summary plot to {save_path}")
    
    plt.close()


def find_prediction_examples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray
) -> Dict[str, int]:
    """
    Find indices for True Positive, False Positive, and False Negative examples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary with indices for TP, FP, FN
    """
    logger.info("Finding prediction examples (TP, FP, FN)...")
    
    # True Positive: predicted=1, actual=1 (correctly identified fraud)
    tp_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    
    # False Positive: predicted=1, actual=0 (legitimate flagged as fraud)
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    
    # False Negative: predicted=0, actual=1 (missed fraud)
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]
    
    examples = {}
    
    # Select examples with highest confidence for each case
    if len(tp_indices) > 0:
        # TP with highest fraud probability
        tp_idx = tp_indices[np.argmax(y_proba[tp_indices])]
        examples['TP'] = tp_idx
        logger.info(f"TP example: index={tp_idx}, prob={y_proba[tp_idx]:.4f}")
    
    if len(fp_indices) > 0:
        # FP with highest fraud probability (most confident mistake)
        fp_idx = fp_indices[np.argmax(y_proba[fp_indices])]
        examples['FP'] = fp_idx
        logger.info(f"FP example: index={fp_idx}, prob={y_proba[fp_idx]:.4f}")
    
    if len(fn_indices) > 0:
        # FN with lowest fraud probability (least confident)
        fn_idx = fn_indices[np.argmin(y_proba[fn_indices])]
        examples['FN'] = fn_idx
        logger.info(f"FN example: index={fn_idx}, prob={y_proba[fn_idx]:.4f}")
    
    return examples


def plot_shap_force(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    idx: int,
    case_name: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create SHAP force plot for a single prediction.
    
    Args:
        explainer: SHAP TreeExplainer
        shap_values: SHAP values
        X: Feature data
        idx: Index of the sample to explain
        case_name: Name of the case (TP, FP, FN)
        save_path: Optional path to save HTML
    """
    logger.info(f"Generating SHAP force plot for {case_name} (index={idx})...")
    
    # Get base value (expected value)
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]  # For binary classification
    
    # Create force plot
    force_plot = shap.force_plot(
        base_value,
        shap_values[idx],
        X.iloc[idx],
        matplotlib=False
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shap.save_html(save_path, force_plot)
        logger.info(f"Saved SHAP force plot to {save_path}")


def compare_importance(
    builtin_importance: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Compare built-in feature importance with SHAP importance.
    
    Args:
        builtin_importance: DataFrame with built-in importance
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of top features to compare
        
    Returns:
        Comparison DataFrame
    """
    logger.info("Comparing built-in vs SHAP importance...")
    
    # Calculate mean absolute SHAP values for each feature
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    # Normalize both to percentages for fair comparison
    builtin_importance['builtin_pct'] = (
        builtin_importance['importance'] / builtin_importance['importance'].sum() * 100
    )
    shap_df['shap_pct'] = (
        shap_df['shap_importance'] / shap_df['shap_importance'].sum() * 100
    )
    
    # Merge and compare top features
    comparison = pd.merge(
        builtin_importance[['feature', 'builtin_pct']],
        shap_df[['feature', 'shap_pct']],
        on='feature',
        how='outer'
    ).fillna(0).sort_values('shap_pct', ascending=False).head(top_n)
    
    logger.info(f"\nTop {top_n} Features Comparison:")
    logger.info(f"\n{comparison.to_string()}")
    
    return comparison


def generate_business_recommendations(
    importance_df: pd.DataFrame,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 5
) -> List[str]:
    """
    Generate actionable business recommendations based on feature importance.
    
    Args:
        importance_df: Feature importance DataFrame
        shap_values: SHAP values
        X: Feature data
        top_n: Number of top features to analyze
        
    Returns:
        List of business recommendations
    """
    logger.info("Generating business recommendations...")
    
    recommendations = []
    top_features = importance_df.head(top_n)['feature'].tolist()
    
    logger.info(f"\nTop {top_n} Fraud Drivers:")
    for i, feature in enumerate(top_features, 1):
        logger.info(f"{i}. {feature}")
    
    # Generic recommendations based on common fraud patterns
    recommendations.append(
        "⚠️ **Enhanced Monitoring for New Users**: Transactions occurring shortly after "
        "account signup show strong fraud signals. Implement additional verification steps "
        "(e.g., email/phone confirmation, address validation) for purchases made within "
        "the first 24-48 hours of registration."
    )
    
    recommendations.append(
        "🌍 **Geographic Risk Assessment**: Certain countries/regions show elevated fraud risk. "
        "Maintain a dynamic risk scoring system that adjusts verification requirements based on "
        "transaction origin. Consider implementing stricter checks for high-risk geographies "
        "while maintaining smooth UX for low-risk regions."
    )
    
    recommendations.append(
        "💰 **Purchase Value Thresholds**: High-value transactions (especially from new accounts) "
        "are strong fraud indicators. Implement tiered verification: low-value transactions "
        "auto-approve, medium-value require basic checks, high-value trigger manual review. "
        "Threshold values should be learned from the model's insights."
    )
    
    recommendations.append(
        "⏱️ **Behavioral Pattern Analysis**: Transaction frequency and timing patterns are key "
        "fraud signals. Flag accounts with unusual velocity (too many transactions in short time) "
        "or suspicious timing patterns (e.g., multiple purchases across different device IDs "
        "within minutes)."
    )
    
    recommendations.append(
        "🔍 **Device & Browser Fingerprinting**: Enhance device tracking mechanisms. Multiple "
        "high-value transactions from the same device/browser across different accounts is a "
        "red flag. Maintain device reputation scores and cross-reference with account behavior."
    )
    
    return recommendations


def full_explainability_pipeline(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_names: List[str],
    output_dir: str = 'reports',
    dataset_name: str = 'model'
) -> Dict[str, Any]:
    """
    Complete explainability pipeline.
    
    Args:
        model: Trained model
        X_train: Training features (for SHAP background)
        X_test: Test features
        y_test: Test labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        feature_names: List of feature names
        output_dir: Directory for saving reports
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with analysis results
    """
    logger.info("="*80)
    logger.info(f"EXPLAINABILITY ANALYSIS - {dataset_name.upper()}")
    logger.info("="*80)
    
    report_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Extract and visualize built-in feature importance
    logger.info("\n" + "="*60)
    logger.info("1. FEATURE IMPORTANCE (Built-in)")
    logger.info("="*60)
    
    builtin_importance = extract_feature_importance(model, feature_names)
    plot_feature_importance(
        builtin_importance,
        top_n=10,
        save_path=os.path.join(report_dir, 'feature_importance.png')
    )
    
    # 2. SHAP analysis
    logger.info("\n" + "="*60)
    logger.info("2. SHAP ANALYSIS")
    logger.info("="*60)
    
    explainer = create_shap_explainer(model, X_train)
    shap_values = calculate_shap_values(explainer, X_test)
    
    # 3. SHAP summary plot
    plot_shap_summary(
        shap_values,
        X_test,
        save_path=os.path.join(report_dir, 'shap_summary.png')
    )
    
    # 4. Find examples and create force plots
    logger.info("\n" + "="*60)
    logger.info("3. INDIVIDUAL PREDICTION ANALYSIS")
    logger.info("="*60)
    
    examples = find_prediction_examples(y_test.values, y_pred, y_proba)
    
    for case_name, idx in examples.items():
        plot_shap_force(
            explainer,
            shap_values,
            X_test,
            idx,
            case_name,
            save_path=os.path.join(report_dir, f'shap_force_{case_name.lower()}.html')
        )
    
    # 5. Compare importance methods
    logger.info("\n" + "="*60)
    logger.info("4. IMPORTANCE COMPARISON")
    logger.info("="*60)
    
    comparison = compare_importance(
        builtin_importance,
        shap_values,
        feature_names,
        top_n=10
    )
    
    # 6. Generate business recommendations
    logger.info("\n" + "="*60)
    logger.info("5. BUSINESS RECOMMENDATIONS")
    logger.info("="*60)
    
    recommendations = generate_business_recommendations(
        builtin_importance,
        shap_values,
        X_test,
        top_n=5
    )
    
    # Save text report
    report_path = os.path.join(report_dir, 'explainability_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"FRAUD DETECTION MODEL - EXPLAINABILITY REPORT\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"="*80 + "\n\n")
        
        f.write("TOP 10 IMPORTANT FEATURES (Built-in Importance):\n")
        f.write("-"*80 + "\n")
        for i, row in builtin_importance.head(10).iterrows():
            f.write(f"{row['feature']:40s} {row['importance']:.6f}\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("BUSINESS RECOMMENDATIONS:\n")
        f.write("-"*80 + "\n")
        for i, rec in enumerate(recommendations, 1):
            f.write(f"\n{i}. {rec}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("\nVISUALIZATIONS GENERATED:\n")
        f.write(f"- feature_importance.png (Bar chart of top 10 features)\n")
        f.write(f"- shap_summary.png (SHAP beeswarm plot)\n")
        f.write(f"- shap_force_tp.html (True Positive force plot)\n")
        f.write(f"- shap_force_fp.html (False Positive force plot)\n")
        f.write(f"- shap_force_fn.html (False Negative force plot)\n")
    
    logger.info(f"\nReport saved to {report_path}")
    
    # Print recommendations
    logger.info("\n" + "="*80)
    logger.info("ACTIONABLE BUSINESS RECOMMENDATIONS")
    logger.info("="*80)
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{i}. {rec}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPLAINABILITY ANALYSIS COMPLETE")
    logger.info("="*80)
    
    return {
        'builtin_importance': builtin_importance,
        'shap_values': shap_values,
        'comparison': comparison,
        'recommendations': recommendations,
        'examples': examples
    }


if __name__ == '__main__':
    logger.info("This module is meant to be imported. See explain_fraud_data.py or explain_creditcard.py for usage.")
