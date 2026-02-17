"""
Tests for metrics module.
"""

import numpy as np
import pytest

from lynxlearn import metrics
from lynxlearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class TestMeanSquaredError:
    """Tests for MSE."""

    def test_perfect_prediction(self):
        """Test MSE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mse = mean_squared_error(y_true, y_pred)
        assert mse == 0.0

    def test_known_error(self):
        """Test MSE with known error."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # Error of 1 for each

        mse = mean_squared_error(y_true, y_pred)
        expected = (1**2 + 1**2 + 1**2) / 3
        assert mse == expected

    def test_alias_mse(self):
        """Test that metrics.mse is alias."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert metrics.mse(y_true, y_pred) == mean_squared_error(y_true, y_pred)


class TestRootMeanSquaredError:
    """Tests for RMSE."""

    def test_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        rmse = root_mean_squared_error(y_true, y_pred)
        assert rmse == 0.0

    def test_relation_to_mse(self):
        """Test that RMSE = sqrt(MSE)."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        assert np.isclose(rmse, np.sqrt(mse))

    def test_alias_rmse(self):
        """Test that metrics.rmse is alias."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert metrics.rmse(y_true, y_pred) == root_mean_squared_error(y_true, y_pred)


class TestMeanAbsoluteError:
    """Tests for MAE."""

    def test_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        mae = mean_absolute_error(y_true, y_pred)
        assert mae == 0.0

    def test_known_error(self):
        """Test MAE with known error."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 5])  # Errors: 1, 1, 2

        mae = mean_absolute_error(y_true, y_pred)
        expected = (1 + 1 + 2) / 3
        assert mae == expected

    def test_alias_mae(self):
        """Test that metrics.mae is alias."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert metrics.mae(y_true, y_pred) == mean_absolute_error(y_true, y_pred)


class TestR2Score:
    """Tests for R² score."""

    def test_perfect_prediction(self):
        """Test R² with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        r2 = r2_score(y_true, y_pred)
        assert r2 == 1.0

    def test_constant_prediction(self):
        """Test R² with constant prediction (predicting mean)."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # Predicting mean

        r2 = r2_score(y_true, y_pred)
        assert r2 == 0.0

    def test_worse_than_mean(self):
        """Test R² when predictions are worse than mean."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([100, 200, 300, 400, 500])  # Very wrong

        r2 = r2_score(y_true, y_pred)
        assert r2 < 0.0

    def test_constant_y(self):
        """Test R² with constant y_true."""
        y_true = np.array([5, 5, 5, 5, 5])
        y_pred = np.array([5, 5, 5, 5, 5])

        r2 = r2_score(y_true, y_pred)
        assert r2 == 1.0  # Perfect prediction of constant

    def test_1d_array_input(self):
        """Test with 1D arrays."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        r2 = r2_score(y_true, y_pred)
        assert 0 < r2 < 1


class TestMetricsConsistency:
    """Tests for consistency across metrics."""

    def test_metrics_ordering(self):
        """Test that better predictions give better metrics."""
        y_true = np.array([1, 2, 3, 4, 5])

        # Worse predictions
        y_pred_worse = np.array([10, 20, 30, 40, 50])

        # Better predictions
        y_pred_better = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

        # Better predictions should have lower error
        assert mean_squared_error(y_true, y_pred_better) < mean_squared_error(
            y_true, y_pred_worse
        )
        assert mean_absolute_error(y_true, y_pred_better) < mean_absolute_error(
            y_true, y_pred_worse
        )

        # Better predictions should have higher R²
        assert r2_score(y_true, y_pred_better) > r2_score(y_true, y_pred_worse)
