"""
Machine Learning pipeline for software defect prediction.
Trains multiple models and returns evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def prepare_data(df: pd.DataFrame):
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
    test_idx = y[y == 0].sample(600, random_state=42).index.append(
                y[y == 1].sample(600, random_state=42).index
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


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'model_name': 'Random Forest',
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'predictions': predictions
    }


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost classifier."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'model_name': 'XGBoost',
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'predictions': predictions
    }


def train_svm_linear(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM with linear kernel."""
    model = SVC(kernel="linear", probability=True, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'model_name': 'SVM (Linear)',
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'predictions': predictions
    }


def train_svm_rbf(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM with RBF kernel."""
    model = SVC(kernel="rbf", probability=True, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'model_name': 'SVM (RBF)',
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'predictions': predictions
    }


def train_svm_sigmoid(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM with sigmoid kernel."""
    model = SVC(kernel="sigmoid", probability=True, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        'model_name': 'SVM (Sigmoid)',
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1_score': f1_score(y_test, predictions),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'predictions': predictions
    }


def train_all_models(csv_path: str):
    """
    Train all models and return results.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with results for all models
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Prepare data
    X_train_sm, y_train_sm, X_test, y_test = prepare_data(df)

    # Train all models
    results = []
    results.append(train_random_forest(X_train_sm, y_train_sm, X_test, y_test))
    results.append(train_xgboost(X_train_sm, y_train_sm, X_test, y_test))
    results.append(train_svm_linear(X_train_sm, y_train_sm, X_test, y_test))
    results.append(train_svm_rbf(X_train_sm, y_train_sm, X_test, y_test))
    results.append(train_svm_sigmoid(X_train_sm, y_train_sm, X_test, y_test))

    # Find best model
    best_model = max(results, key=lambda x: x['accuracy'])

    return {
        'models': results,
        'best_model': best_model['model_name'],
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train_sm),
            'test_samples': len(X_test),
            'defective_in_test': int((y_test == 1).sum()),
            'non_defective_in_test': int((y_test == 0).sum())
        }
    }
