"""
Tests for neural network loss functions.
"""

import numpy as np
import pytest

from lynxlearn.neural_network.losses import (
    MAE,
    MSE,
    BaseLoss,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
)


class TestMeanSquaredError:
    """Tests for MeanSquaredError loss function."""

    def test_initialization(self):
        """Test MSE initialization."""
        loss = MeanSquaredError()
        assert loss.name == "mse"
        assert loss.reduction == "mean"

    def test_compute_basic(self):
        """Test basic MSE computation."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])

        result = loss.compute(y_true, y_pred)
        assert result == 0.0

    def test_compute_with_error(self):
        """Test MSE computation with prediction error."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[2.0], [3.0], [4.0]])

        # MSE = mean((1-2)^2 + (2-3)^2 + (3-4)^2) = mean(1 + 1 + 1) = 1.0
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 1.0)

    def test_compute_1d_arrays(self):
        """Test MSE with 1D arrays."""
        loss = MeanSquaredError()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])

        # MSE = mean((0.5)^2 + (0.5)^2 + (0.5)^2) = 0.25
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 0.25)

    def test_compute_multidimensional(self):
        """Test MSE with multi-dimensional output."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = loss.compute(y_true, y_pred)
        assert result == 0.0

    def test_compute_sum_reduction(self):
        """Test MSE with sum reduction."""
        loss = MeanSquaredError(reduction="sum")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])

        # Sum = 1 + 1 + 1 = 3
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 3.0)

    def test_compute_none_reduction(self):
        """Test MSE with no reduction."""
        loss = MeanSquaredError(reduction="none")
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])

        result = loss.compute(y_true, y_pred)
        assert result.shape == (3,)
        assert np.allclose(result, [1.0, 1.0, 1.0])

    def test_gradient_basic(self):
        """Test MSE gradient computation."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[2.0], [3.0], [4.0]])

        # gradient = 2 * (y_pred - y_true) / n = 2 * [1, 1, 1] / 3
        grad = loss.gradient(y_true, y_pred)
        expected = 2.0 * np.array([[1.0], [1.0], [1.0]]) / 3.0
        assert np.allclose(grad, expected)

    def test_gradient_shape(self):
        """Test that gradient has correct shape."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.5, 2.5], [3.5, 4.5]])

        grad = loss.gradient(y_true, y_pred)
        assert grad.shape == y_true.shape

    def test_shape_mismatch_error(self):
        """Test that shape mismatch raises error."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError):
            loss.compute(y_true, y_pred)

    def test_call_method(self):
        """Test __call__ method."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.5], [2.5]])

        result = loss(y_true, y_pred)
        assert np.isclose(result, 0.25)

    def test_call_with_gradient(self):
        """Test __call__ with return_grad=True."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.5], [2.5]])

        loss_val, grad = loss(y_true, y_pred, return_grad=True)
        assert np.isclose(loss_val, 0.25)
        assert grad.shape == y_true.shape

    def test_repr(self):
        """Test string representation."""
        loss = MeanSquaredError()
        assert "MeanSquaredError" in repr(loss)

    def test_alias_mse(self):
        """Test MSE alias."""
        assert MSE is MeanSquaredError


class TestMeanAbsoluteError:
    """Tests for MeanAbsoluteError loss function."""

    def test_initialization(self):
        """Test MAE initialization."""
        loss = MeanAbsoluteError()
        assert loss.name == "mae"
        assert loss.reduction == "mean"

    def test_compute_basic(self):
        """Test basic MAE computation."""
        loss = MeanAbsoluteError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])

        result = loss.compute(y_true, y_pred)
        assert result == 0.0

    def test_compute_with_error(self):
        """Test MAE computation with prediction error."""
        loss = MeanAbsoluteError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[2.0], [3.0], [4.0]])

        # MAE = mean(|1-2| + |2-3| + |3-4|) = mean(1 + 1 + 1) = 1.0
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 1.0)

    def test_gradient_basic(self):
        """Test MAE gradient computation."""
        loss = MeanAbsoluteError()
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[2.0], [3.0], [4.0]])

        # gradient = sign(y_pred - y_true) / n = [1, 1, 1] / 3
        grad = loss.gradient(y_true, y_pred)
        expected = np.array([[1.0], [1.0], [1.0]]) / 3.0
        assert np.allclose(grad, expected)

    def test_gradient_negative_error(self):
        """Test MAE gradient with negative errors."""
        loss = MeanAbsoluteError()
        y_true = np.array([[2.0], [3.0], [4.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])

        # gradient = sign(y_pred - y_true) / n = [-1, -1, -1] / 3
        grad = loss.gradient(y_true, y_pred)
        expected = np.array([[-1.0], [-1.0], [-1.0]]) / 3.0
        assert np.allclose(grad, expected)

    def test_repr(self):
        """Test string representation."""
        loss = MeanAbsoluteError()
        assert "MeanAbsoluteError" in repr(loss)

    def test_alias_mae(self):
        """Test MAE alias."""
        assert MAE is MeanAbsoluteError


class TestHuberLoss:
    """Tests for HuberLoss loss function."""

    def test_initialization(self):
        """Test HuberLoss initialization."""
        loss = HuberLoss()
        assert loss.name == "huber"
        assert loss.delta == 1.0
        assert loss.reduction == "mean"

    def test_initialization_custom_delta(self):
        """Test HuberLoss with custom delta."""
        loss = HuberLoss(delta=0.5)
        assert loss.delta == 0.5

    def test_invalid_delta(self):
        """Test that invalid delta raises error."""
        with pytest.raises(ValueError):
            HuberLoss(delta=0.0)

        with pytest.raises(ValueError):
            HuberLoss(delta=-1.0)

    def test_compute_small_error(self):
        """Test HuberLoss with small errors (quadratic region)."""
        loss = HuberLoss(delta=1.0)
        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.2], [2.2]])

        # Error = 0.2, which is < delta=1.0
        # Loss = 0.5 * 0.2^2 = 0.02
        result = loss.compute(y_true, y_pred)
        expected = 0.5 * (0.2**2)
        assert np.isclose(result, expected)

    def test_compute_large_error(self):
        """Test HuberLoss with large errors (linear region)."""
        loss = HuberLoss(delta=1.0)
        y_true = np.array([[0.0], [0.0]])
        y_pred = np.array([[2.0], [3.0]])

        # Errors = 2.0 and 3.0, both > delta=1.0
        # Loss = delta * |error| - 0.5 * delta^2
        # For error=2: 1*2 - 0.5*1 = 1.5
        # For error=3: 1*3 - 0.5*1 = 2.5
        # Mean = 2.0
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 2.0)

    def test_compute_mixed_errors(self):
        """Test HuberLoss with mixed small and large errors."""
        loss = HuberLoss(delta=1.0)
        y_true = np.array([[0.0], [0.0]])
        y_pred = np.array([[0.5], [2.0]])

        # Error=0.5 < delta: loss = 0.5 * 0.25 = 0.125
        # Error=2.0 > delta: loss = 1*2 - 0.5*1 = 1.5
        # Mean = 0.8125
        result = loss.compute(y_true, y_pred)
        assert np.isclose(result, 0.8125)

    def test_gradient_small_error(self):
        """Test HuberLoss gradient with small error."""
        loss = HuberLoss(delta=1.0)
        y_true = np.array([[0.0]])
        y_pred = np.array([[0.5]])

        # Error = 0.5 < delta, gradient = error/n
        grad = loss.gradient(y_true, y_pred)
        expected = np.array([[0.5]])
        assert np.allclose(grad, expected)

    def test_gradient_large_error(self):
        """Test HuberLoss gradient with large error."""
        loss = HuberLoss(delta=1.0)
        y_true = np.array([[0.0]])
        y_pred = np.array([[2.0]])

        # Error = 2.0 > delta, gradient = delta * sign(error) / n
        grad = loss.gradient(y_true, y_pred)
        expected = np.array([[1.0]])
        assert np.allclose(grad, expected)

    def test_repr(self):
        """Test string representation."""
        loss = HuberLoss(delta=0.5)
        assert "HuberLoss" in repr(loss)
        assert "0.5" in repr(loss)


class TestBaseLoss:
    """Tests for BaseLoss abstract class."""

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError):
            MeanSquaredError(reduction="invalid")

    def test_get_config(self):
        """Test get_config method."""
        loss = MeanSquaredError(reduction="sum")
        config = loss.get_config()

        assert config["reduction"] == "sum"
        assert config["name"] == "mse"

    def test_from_config(self):
        """Test from_config class method."""
        config = {"reduction": "sum", "name": "mse"}
        loss = MeanSquaredError.from_config(config)

        assert loss.reduction == "sum"
        assert loss.name == "mse"


class TestLossNumericalStability:
    """Tests for numerical stability of loss functions."""

    def test_mse_large_values(self):
        """Test MSE with large values."""
        loss = MeanSquaredError()
        y_true = np.array([[1e6], [2e6]])
        y_pred = np.array([[1e6 + 1], [2e6 + 1]])

        result = loss.compute(y_true, y_pred)
        assert np.isfinite(result)

    def test_mse_small_values(self):
        """Test MSE with very small values."""
        loss = MeanSquaredError()
        y_true = np.array([[1e-10], [2e-10]])
        y_pred = np.array([[2e-10], [3e-10]])

        result = loss.compute(y_true, y_pred)
        assert np.isfinite(result)

    def test_mse_with_nan(self):
        """Test MSE behavior with NaN values."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [np.nan]])
        y_pred = np.array([[2.0], [3.0]])

        result = loss.compute(y_true, y_pred)
        assert np.isnan(result)

    def test_mse_with_inf(self):
        """Test MSE behavior with infinite values."""
        loss = MeanSquaredError()
        y_true = np.array([[1.0], [np.inf]])
        y_pred = np.array([[2.0], [3.0]])

        result = loss.compute(y_true, y_pred)
        assert np.isinf(result)
