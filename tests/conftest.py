"""
Pytest configuration and shared fixtures.
"""

import numpy as np
import pytest


@pytest.fixture
def simple_regression_data():
    """Generate simple 1D regression data."""
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1)
    true_weights = 2.5
    true_bias = 5.0
    noise = np.random.randn(n_samples) * 0.5
    y = X.flatten() * true_weights + true_bias + noise
    return X, y, true_weights, true_bias


@pytest.fixture
def multi_feature_data():
    """Generate multi-feature regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -2.0, 0.5, 3.0, -1.0])
    true_bias = 2.0
    noise = np.random.randn(n_samples) * 0.3
    y = X @ true_weights + true_bias + noise
    return X, y, true_weights, true_bias


@pytest.fixture
def train_test_data(simple_regression_data):
    """Generate train/test split data."""
    from lynxlearn.model_selection import train_test_split

    X, y, true_weights, true_bias = simple_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, true_weights, true_bias
