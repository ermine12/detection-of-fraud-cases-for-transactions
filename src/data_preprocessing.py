import os
import logging
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_FRAUD_COLUMNS: List[str] = [
    'user_id', 'signup_time', 'purchase_time', 'purchase_value', 'device_id',
    'source', 'browser', 'sex', 'age', 'ip_address', 'class'
]

REQUIRED_IP_COLUMNS: List[str] = [
    'lower_bound_ip_address', 'upper_bound_ip_address', 'country'
]

def load_data(
    fraud_path: str = 'data/raw/Fraud_Data.csv',
    ip_country_path: str = 'data/raw/IpAddress_to_Country.csv',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from CSV files.

    Inputs:
    - fraud_path: path to Fraud_Data.csv
    - ip_country_path: path to IpAddress_to_Country.csv

    Outputs:
    - Tuple of (fraud_data, ip_country) DataFrames

    Raises:
    - FileNotFoundError with informative message if a file is missing
    - ValueError if loaded objects are empty
    """
    logger.info(f"Loading fraud data from: {fraud_path}")
    logger.info(f"Loading IP-country data from: {ip_country_path}")
    
    try:
        if not os.path.exists(fraud_path):
            logger.error(f"Fraud data file not found: {fraud_path}")
            raise FileNotFoundError(f"Fraud data file not found: {fraud_path}")
        if not os.path.exists(ip_country_path):
            logger.error(f"IP-country file not found: {ip_country_path}")
            raise FileNotFoundError(f"IP-country file not found: {ip_country_path}")

        fraud_data = pd.read_csv(fraud_path)
        ip_country = pd.read_csv(ip_country_path)
        logger.info(f"Loaded {len(fraud_data)} fraud records and {len(ip_country)} IP ranges")
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to read CSV files: {e}")
        raise RuntimeError(f"Failed to read CSV files: {e}")

    if fraud_data is None or fraud_data.empty:
        raise ValueError("Fraud data is empty after loading.")
    if ip_country is None or ip_country.empty:
        raise ValueError("IP-country data is empty after loading.")

    return fraud_data, ip_country

def _validate_required_columns(df: pd.DataFrame, required: List[str], df_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def clean_data(fraud_data: pd.DataFrame, ip_country: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Data cleaning and basic validation.

    Inputs:
    - fraud_data: raw fraud transactions
    - ip_country: IP address ranges mapped to countries

    Outputs:
    - Cleaned (fraud_data, ip_country)

    Validations performed:
    - Required columns existence
    - Type conversions for datetime and integer IPs
    - Basic missing values handling and duplicate removal

    Raises:
    - ValueError for missing columns or invalid types
    """
    logger.info("Validating required columns...")
    _validate_required_columns(fraud_data, REQUIRED_FRAUD_COLUMNS, 'fraud_data')
    _validate_required_columns(ip_country, REQUIRED_IP_COLUMNS, 'ip_country')
    logger.info("Column validation passed")

    # Report missing values
    logger.info("Checking for missing values...")
    missing_fraud = fraud_data.isnull().sum()
    missing_ip = ip_country.isnull().sum()
    if missing_fraud.any():
        logger.warning(f'Missing values in Fraud Data:\n{missing_fraud[missing_fraud > 0]}')
    if missing_ip.any():
        logger.warning(f'Missing values in IP Country:\n{missing_ip[missing_ip > 0]}')

    # Drop rows with missing values in critical fields
    logger.info("Dropping rows with missing critical values...")
    fraud_data = fraud_data.dropna(subset=['signup_time', 'purchase_time', 'purchase_value', 'ip_address', 'class'])
    ip_country = ip_country.dropna(subset=['lower_bound_ip_address', 'upper_bound_ip_address', 'country'])
    logger.info(f"After dropping NaNs: {len(fraud_data)} fraud records, {len(ip_country)} IP ranges")

    # Remove duplicates
    if fraud_data.duplicated().any():
        dup_count = fraud_data.duplicated().sum()
        logger.info(f'Found {dup_count} duplicates in Fraud Data, removing...')
        fraud_data = fraud_data.drop_duplicates()

    # Convert data types with safety
    logger.info("Converting data types...")
    for dt_col in ['signup_time', 'purchase_time']:
        try:
            fraud_data[dt_col] = pd.to_datetime(fraud_data[dt_col], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to convert {dt_col} to datetime: {e}")
            raise ValueError(f"Failed to convert {dt_col} to datetime: {e}")
    # Ensure datetimes were parsed
    if fraud_data[['signup_time', 'purchase_time']].isnull().any().any():
        raise ValueError('Datetime parsing produced NaT values; check input formats for signup_time/purchase_time.')

    # Numeric conversions
    for col in ['ip_address', 'age']:
        if col in fraud_data.columns:
            fraud_data[col] = pd.to_numeric(fraud_data[col], errors='coerce')
    if fraud_data[['ip_address']].isnull().any().any():
        raise ValueError('ip_address contains non-numeric values after coercion.')
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(np.int64)

    for col in ['lower_bound_ip_address', 'upper_bound_ip_address']:
        ip_country[col] = pd.to_numeric(ip_country[col], errors='coerce')
        if ip_country[col].isnull().any():
            raise ValueError(f"{col} contains non-numeric values after coercion in ip_country.")
        ip_country[col] = ip_country[col].astype(np.int64)

    return fraud_data, ip_country

def merge_geolocation(fraud_data: pd.DataFrame, ip_country: pd.DataFrame) -> pd.DataFrame:
    """Annotate fraud_data with country based on IP ranges.

    Inputs:
    - fraud_data: cleaned transactions (must include ip_address)
    - ip_country: cleaned IP ranges with lower/upper bounds and country

    Outputs:
    - fraud_data with a new 'country' column

    Raises:
    - ValueError if required columns are missing
    """
    logger.info("Merging geolocation data...")
    _validate_required_columns(fraud_data, ['ip_address'], 'fraud_data')
    _validate_required_columns(ip_country, REQUIRED_IP_COLUMNS, 'ip_country')

    # Sort ip_country for efficient lookup
    ip_country = ip_country.sort_values('lower_bound_ip_address').reset_index(drop=True)
    logger.info(f"Processing {len(fraud_data)} IP addresses...")

    def find_country(ip: int) -> str:
        mask = (ip_country['lower_bound_ip_address'] <= ip) & (ip_country['upper_bound_ip_address'] >= ip)
        matches = ip_country.loc[mask]
        if not matches.empty:
            return str(matches['country'].iloc[0])
        return 'Unknown'

    fraud_data = fraud_data.copy()
    fraud_data['country'] = fraud_data['ip_address'].apply(find_country)
    logger.info(f"Geolocation merge complete. Unknown IPs: {(fraud_data['country'] == 'Unknown').sum()}")
    return fraud_data

def feature_engineering(fraud_data: pd.DataFrame) -> pd.DataFrame:
    """Create derived features.

    Inputs:
    - fraud_data: transactions with signup_time, purchase_time, user_id

    Outputs:
    - fraud_data augmented with engineered features

    Raises:
    - ValueError if required columns are missing
    """
    logger.info("Engineering features...")
    _validate_required_columns(fraud_data, ['signup_time', 'purchase_time', 'user_id'], 'fraud_data')

    fraud_data = fraud_data.copy()
    # Time-based features
    fraud_data['hour_of_day'] = fraud_data['purchase_time'].dt.hour
    fraud_data['day_of_week'] = fraud_data['purchase_time'].dt.dayofweek
    fraud_data['time_since_signup'] = (
        (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds() / 3600.0
    )  # hours

    # Transaction frequency per user
    fraud_data = fraud_data.sort_values(['user_id', 'purchase_time'])
    fraud_data['transaction_count'] = fraud_data.groupby('user_id').cumcount() + 1

    logger.info("Feature engineering complete. Created: hour_of_day, day_of_week, time_since_signup, transaction_count")
    return fraud_data

def transform_data(fraud_data: pd.DataFrame):
    """Scale numeric features and encode categoricals.

    Inputs:
    - fraud_data: dataset containing numeric and categorical columns

    Outputs:
    - transformed fraud_data, fitted scaler, fitted encoder

    Raises:
    - ValueError if required columns are missing or non-numeric where expected
    """
    numerical_cols = ['purchase_value', 'age', 'time_since_signup', 'transaction_count']
    categorical_cols = ['source', 'browser', 'sex', 'country']

    logger.info("Transforming features...")
    # Validate columns
    _validate_required_columns(fraud_data, numerical_cols, 'fraud_data (numerical)')
    _validate_required_columns(fraud_data, categorical_cols, 'fraud_data (categorical)')

    # Numeric types
    for col in numerical_cols:
        fraud_data[col] = pd.to_numeric(fraud_data[col], errors='coerce')
    if fraud_data[numerical_cols].isnull().any().any():
        raise ValueError('Numerical columns contain NaN after coercion; check inputs.')

    logger.info(f"Scaling numerical features: {numerical_cols}")
    scaler = StandardScaler()
    fraud_data[numerical_cols] = scaler.fit_transform(fraud_data[numerical_cols])

    logger.info(f"One-hot encoding categorical features: {categorical_cols}")
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded = encoder.fit_transform(fraud_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols), index=fraud_data.index)
    fraud_data = pd.concat([fraud_data.drop(categorical_cols, axis=1), encoded_df], axis=1)

    logger.info(f"Transformation complete. Final shape: {fraud_data.shape}")
    return fraud_data, scaler, encoder

def handle_imbalance(X: pd.DataFrame, y: pd.Series):
    """Apply SMOTE to handle class imbalance.

    Inputs:
    - X: feature matrix
    - y: target labels (binary)

    Outputs:
    - X_resampled, y_resampled, fitted SMOTE instance

    Raises:
    - ValueError if y is not binary or has insufficient minority samples
    """
    logger.info("Handling class imbalance with SMOTE...")
    if not isinstance(y, (pd.Series, np.ndarray)):
        logger.error('y must be a pandas Series or NumPy array')
        raise ValueError('y must be a pandas Series or NumPy array')
    y_series = pd.Series(y)
    if y_series.nunique() != 2:
        logger.error('y must be binary for SMOTE')
        raise ValueError('y must be binary for SMOTE')

    logger.info(f'Original class distribution: {y_series.value_counts().to_dict()}')

    smote = SMOTE(random_state=42)
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y_series)
    except ValueError as e:
        logger.error(f"SMOTE failed: {e}")
        raise ValueError(f"SMOTE failed: {e}")

    logger.info(f'After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}')
    return X_resampled, y_resampled, smote

def main():
    """End-to-end preprocessing pipeline entrypoint.

    Loads data, validates and cleans, enriches with geolocation, engineers features,
    transforms features, applies SMOTE, and saves processed dataset to CSV.
    """
    logger.info("="*60)
    logger.info("Starting preprocessing pipeline")
    logger.info("="*60)
    
    try:
        # Load
        fraud_data, ip_country = load_data()

        # Clean
        fraud_data, ip_country = clean_data(fraud_data, ip_country)

        # Geolocation
        fraud_data = merge_geolocation(fraud_data, ip_country)

        # Feature engineering
        fraud_data = feature_engineering(fraud_data)

        # Drop unnecessary columns (if present)
        cols_to_drop = ['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address']
        cols_present = [c for c in cols_to_drop if c in fraud_data.columns]
        fraud_data = fraud_data.drop(cols_present, axis=1)

        # Split features and target
        if 'class' not in fraud_data.columns:
            raise ValueError("Target column 'class' not found after preprocessing.")
        X = fraud_data.drop('class', axis=1)
        y = fraud_data['class']

        # Transform
        X_transformed, scaler, encoder = transform_data(X.copy())

        # Handle imbalance
        X_final, y_final, smote = handle_imbalance(X_transformed, y)

        # Save processed data
        output_path = 'data/processed/fraud_data_processed.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processed_df = pd.concat([X_final, y_final.rename('class')], axis=1)
        processed_df.to_csv(output_path, index=False)
        logger.info(f'Processed data saved to {output_path}')
        logger.info(f'Final dataset shape: {processed_df.shape}')
    except Exception as e:
        # Provide informative error and non-zero exit for CLI context
        msg = f"Preprocessing failed: {e}"
        logger.error(msg)
        raise


if __name__ == '__main__':
    main()