"""
Machine Learning pipeline for software defect prediction.
XGBoost-only implementation with model persistence.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Model storage directory
MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)


def prepare_training_data(df: pd.DataFrame):
    """
    Prepare training and test datasets with SMOTE oversampling.

    Args:
        df: Input dataframe with features and 'defect' target column

    Returns:
        Tuple of (X_train_sm, y_train_sm, X_test, y_test)
    """
    X = df.drop(columns=['defect'])
    y = df['defect']

    # Create stratified test set (600 samples from each class)
    test_idx = y[y == 0].sample(min(600, (y == 0).sum()), random_state=42).index.union(
                y[y == 1].sample(min(600, (y == 1).sum()), random_state=42).index
               )

    train_idx = y.index.difference(test_idx)

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    # Shuffle datasets
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    # Apply SMOTE to balance training data
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    return X_train_sm, y_train_sm, X_test, y_test


def train_xgboost_model(csv_path: str):
    """
    Train XGBoost model and save it.

    Args:
        csv_path: Path to the CSV file with labeled data

    Returns:
        Dictionary with training results and model path
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_training_data(df)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"xgboost_model_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename

    joblib.dump(model, model_path)

    return {
        'model_name': 'XGBoost',
        'model_path': str(model_path),
        'model_filename': model_filename,
        'timestamp': timestamp,
        'accuracy': float(accuracy_score(y_test, predictions)),
        'precision': float(precision_score(y_test, predictions)),
        'recall': float(recall_score(y_test, predictions)),
        'f1_score': float(f1_score(y_test, predictions)),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'defective_in_test': int((y_test == 1).sum()),
            'non_defective_in_test': int((y_test == 0).sum())
        }
    }


def load_model(model_path: str):
    """
    Load a saved model.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded XGBoost model
    """
    return joblib.load(model_path)


def get_available_models():
    """
    Get list of all saved models.

    Returns:
        List of dictionaries with model info
    """
    models = []
    for model_file in MODELS_DIR.glob("xgboost_model_*.joblib"):
        # Extract timestamp from filename
        timestamp_str = model_file.stem.replace("xgboost_model_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            models.append({
                'filename': model_file.name,
                'path': str(model_file),
                'timestamp': timestamp_str,
                'created': timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
        except ValueError:
            continue

    # Sort by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models


def predict_with_model(model_path: str, csv_path: str, output_path: str = None):
    """
    Make predictions on unlabeled data using a saved model.

    Args:
        model_path: Path to the saved model
        csv_path: Path to CSV file with unlabeled data (no 'defect' column)
        output_path: Optional path to save predictions CSV

    Returns:
        Dictionary with predictions and statistics
    """
    # Load model
    model = load_model(model_path)

    # Load data
    df = pd.read_csv(csv_path)

    # Check if 'defect' column exists and remove it if present
    has_labels = 'defect' in df.columns
    if has_labels:
        df = df.drop(columns=['defect'])

    # Make predictions
    predictions = model.predict(df)
    prediction_proba = model.predict_proba(df)

    # Create results dataframe
    results_df = df.copy()
    results_df['predicted_defect'] = predictions
    results_df['defect_probability'] = prediction_proba[:, 1]
    results_df['non_defect_probability'] = prediction_proba[:, 0]

    # Save to CSV if output path provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = MODELS_DIR.parent / "predictions" / f"predictions_{timestamp}.csv"
        output_path.parent.mkdir(exist_ok=True)

    results_df.to_csv(output_path, index=False)

    # Calculate statistics
    num_defective = int((predictions == 1).sum())
    num_non_defective = int((predictions == 0).sum())

    return {
        'total_samples': len(predictions),
        'predicted_defective': num_defective,
        'predicted_non_defective': num_non_defective,
        'defect_rate': float(num_defective / len(predictions) * 100),
        'predictions': predictions.tolist(),
        'probabilities': prediction_proba.tolist(),
        'output_file': str(output_path),
        'output_filename': Path(output_path).name
    }
