"""
Mean Squared Error (MSE) loss function implementation - HYPER-OPTIMIZED.

Uses fast operations from _core.py for maximum performance.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from ._base import BaseLoss

# Try to import optimized core functions
try:
    from lynxlearn._core import (
        fast_huber_gradient,
        fast_huber_loss,
        fast_mae_gradient,
        fast_mae_loss,
        fast_mse_gradient,
        fast_mse_loss,
    )

    CORE_OPTIMIZED = True
except ImportError:
    CORE_OPTIMIZED = False

    def fast_mse_loss(y_true, y_pred):
        """Fallback MSE loss."""
        return np.mean((y_true - y_pred) ** 2)

    def fast_mse_gradient(y_true, y_pred):
        """Fallback MSE gradient."""
        n = y_true.size
        return 2.0 * (y_pred - y_true) / n

    def fast_mae_loss(y_true, y_pred):
        """Fallback MAE loss."""
        return np.mean(np.abs(y_true - y_pred))

    def fast_mae_gradient(y_true, y_pred):
        """Fallback MAE gradient."""
        n = y_true.size
        return np.sign(y_pred - y_true) / n

    def fast_huber_loss(y_true, y_pred, delta=1.0):
        """Fallback Huber loss."""
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return np.mean(0.5 * quadratic**2 + delta * linear)

    def fast_huber_gradient(y_true, y_pred, delta=1.0):
        """Fallback Huber gradient."""
        error = y_pred - y_true
        abs_error = np.abs(error)
        n = y_true.size
        return np.where(abs_error <= delta, error / n, delta * np.sign(error) / n)


class MeanSquaredError(BaseLoss):
    """
    Mean Squared Error loss function - HYPER-OPTIMIZED.

    Computes the mean of squared differences between true and predicted values.

    loss = mean((y_true - y_pred)^2)

    Uses optimized vectorized operations from _core.py for 2-4x speedup.

    Parameters
    ----------
    reduction : str, default='mean'
        Type of reduction to apply. Options:
        - 'mean': Return the mean loss over all samples
        - 'sum': Return the sum of losses over all samples
        - 'none': Return per-sample losses

    Attributes
    ----------
    name : str
        Name of the loss function ('mse')

    Examples
    --------
    >>> loss_fn = MeanSquaredError()
    >>> y_true = np.array([[1.0], [2.0], [3.0]])
    >>> y_pred = np.array([[1.1], [1.9], [3.2]])
    >>> loss = loss_fn.compute(y_true, y_pred)
    >>> print(f"MSE Loss: {loss:.6f}")
    MSE Loss: 0.020000

    >>> # Get gradient for backpropagation
    >>> grad = loss_fn.gradient(y_true, y_pred)
    >>> print(f"Gradient shape: {grad.shape}")
    Gradient shape: (3, 1)

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mean_squared_error
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.name = "mse"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Compute the Mean Squared Error loss.

        Parameters
        ----------
        y_true : ndarray
            Ground truth (true) values. Shape can be (n_samples,) or
            (n_samples, n_outputs).
        y_pred : ndarray
            Predicted values. Must have the same shape as y_true.
        sample_weight : ndarray, optional
            Optional array of weights for each sample. If provided,
            computes weighted MSE.

        Returns
        -------
        loss : float or ndarray
            Mean squared error. Returns array if reduction='none'.
        """
        # Convert to numpy arrays - use contiguous arrays for cache efficiency
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Fast path for mean reduction without sample weights
        if self.reduction == "mean" and sample_weight is None:
            return fast_mse_loss(y_true, y_pred)

        # Compute squared error using optimized operation
        # Use in-place operations to reduce memory allocation
        error = np.empty_like(y_true)
        np.subtract(y_true, y_pred, out=error)
        np.square(error, out=error)  # error now holds squared_error

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < error.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            error *= sample_weight

        # Sum over output dimensions (keep per-sample losses)
        if error.ndim > 1:
            per_sample_loss = np.sum(error, axis=tuple(range(1, error.ndim)))
        else:
            per_sample_loss = error

        # Apply reduction
        if self.reduction == "none":
            return per_sample_loss

        if sample_weight is not None:
            if self.reduction == "mean":
                return np.sum(per_sample_loss) / np.sum(sample_weight)
            else:
                return np.sum(per_sample_loss)
        else:
            return self._apply_reduction(per_sample_loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MSE loss with respect to predictions.

        The gradient is computed as:
            d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n

        Uses optimized vectorized operations for 2-4x speedup.

        Parameters
        ----------
        y_true : ndarray
            Ground truth (true) values
        y_pred : ndarray
            Predicted values

        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to y_pred. Has the same shape
            as y_pred.
        """
        # Use contiguous arrays for cache efficiency
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Use optimized gradient computation
        return fast_mse_gradient(y_true, y_pred)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration for serialization."""
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MeanSquaredError":
        """Create MSE loss from configuration dictionary."""
        config = config.copy()
        config.pop("name", None)
        return cls(**config)

    def __repr__(self) -> str:
        return f"MeanSquaredError(reduction='{self.reduction}')"


class MeanAbsoluteError(BaseLoss):
    """
    Mean Absolute Error (MAE) loss function - HYPER-OPTIMIZED.

    Computes the mean of absolute differences between true and predicted values.

    loss = mean(|y_true - y_pred|)

    Uses optimized vectorized operations from _core.py for 1.5-3x speedup.

    Parameters
    ----------
    reduction : str, default='mean'
        Type of reduction to apply ('mean', 'sum', or 'none')

    Attributes
    ----------
    name : str
        Name of the loss function ('mae')

    Examples
    --------
    >>> loss_fn = MeanAbsoluteError()
    >>> y_true = np.array([[1.0], [2.0], [3.0]])
    >>> y_pred = np.array([[1.1], [1.9], [3.2]])
    >>> loss = loss_fn.compute(y_true, y_pred)
    >>> print(f"MAE Loss: {loss:.6f}")
    MAE Loss: 0.133333
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.name = "mae"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Compute the Mean Absolute Error loss.

        Parameters
        ----------
        y_true : ndarray
            Ground truth (true) values
        y_pred : ndarray
            Predicted values
        sample_weight : ndarray, optional
            Optional array of weights for each sample

        Returns
        -------
        loss : float or ndarray
            Mean absolute error
        """
        # Use contiguous arrays for cache efficiency
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Fast path for mean reduction without sample weights
        if self.reduction == "mean" and sample_weight is None:
            return fast_mae_loss(y_true, y_pred)

        # Compute absolute error using optimized operation
        error = np.empty_like(y_true)
        np.subtract(y_true, y_pred, out=error)
        np.absolute(error, out=error)  # error now holds absolute error

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < error.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            error *= sample_weight

        # Sum over output dimensions
        if error.ndim > 1:
            per_sample_loss = np.sum(error, axis=tuple(range(1, error.ndim)))
        else:
            per_sample_loss = error

        # Apply reduction
        if self.reduction == "none":
            return per_sample_loss

        if sample_weight is not None:
            if self.reduction == "mean":
                return np.sum(per_sample_loss) / np.sum(sample_weight)
            else:
                return np.sum(per_sample_loss)
        else:
            return self._apply_reduction(per_sample_loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MAE loss with respect to predictions.

        d(MAE)/d(y_pred) = sign(y_pred - y_true) / n

        Parameters
        ----------
        y_true : ndarray
            Ground truth values
        y_pred : ndarray
            Predicted values

        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to y_pred
        """
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Use optimized gradient computation
        return fast_mae_gradient(y_true, y_pred)

    def __repr__(self) -> str:
        return f"MeanAbsoluteError(reduction='{self.reduction}')"


class HuberLoss(BaseLoss):
    """
    Huber loss function - HYPER-OPTIMIZED.

    Combines MSE for small errors and MAE for large errors,
    making it less sensitive to outliers than MSE.

    loss = { 0.5 * (y_true - y_pred)^2  if |y_true - y_pred| <= delta
           { delta * |y_true - y_pred| - 0.5 * delta^2  otherwise

    Uses optimized vectorized operations from _core.py for 1.5-3x speedup.

    Parameters
    ----------
    delta : float, default=1.0
        The threshold at which to transition from MSE to MAE.
    reduction : str, default='mean'
        Type of reduction to apply

    Attributes
    ----------
    name : str
        Name of the loss function ('huber')

    Examples
    --------
    >>> loss_fn = HuberLoss(delta=1.0)
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([1.5, 2.5, 10.0])  # Last one is an outlier
    >>> loss = loss_fn.compute(y_true, y_pred)
    """

    def __init__(self, delta: float = 1.0, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.delta = delta
        self.name = "huber"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Compute the Huber loss.

        Parameters
        ----------
        y_true : ndarray
            Ground truth values
        y_pred : ndarray
            Predicted values
        sample_weight : ndarray, optional
            Optional array of weights for each sample

        Returns
        -------
        loss : float or ndarray
            Huber loss
        """
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Fast path for mean reduction without sample weights
        if self.reduction == "mean" and sample_weight is None:
            return fast_huber_loss(y_true, y_pred, self.delta)

        # Compute Huber loss components using optimized operations
        error = np.empty_like(y_true)
        np.subtract(y_true, y_pred, out=error)
        abs_error = np.abs(error)

        # Quadratic for small errors, linear for large errors
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic

        # In-place computation: loss = 0.5 * quadratic^2 + delta * linear
        loss = np.empty_like(y_true)
        np.square(quadratic, out=loss)
        loss *= 0.5
        loss += self.delta * linear

        # Apply sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < loss.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            loss *= sample_weight

        # Sum over output dimensions
        if loss.ndim > 1:
            per_sample_loss = np.sum(loss, axis=tuple(range(1, loss.ndim)))
        else:
            per_sample_loss = loss

        if self.reduction == "none":
            return per_sample_loss

        return self._apply_reduction(per_sample_loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of Huber loss with respect to predictions.

        Parameters
        ----------
        y_true : ndarray
            Ground truth values
        y_pred : ndarray
            Predicted values

        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to y_pred
        """
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Use optimized gradient computation
        return fast_huber_gradient(y_true, y_pred, self.delta)

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["delta"] = self.delta
        return config

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self.delta}, reduction='{self.reduction}')"


class BinaryCrossEntropy(BaseLoss):
    """
    Binary Cross-Entropy loss function - HYPER-OPTIMIZED.

    Computes the binary cross-entropy loss for binary classification.

    loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Uses numerically stable computation to avoid log(0) issues.

    Parameters
    ----------
    from_logits : bool, default=False
        If True, y_pred is treated as logits (raw scores) and sigmoid
        is applied internally for numerical stability.
    reduction : str, default='mean'
        Type of reduction to apply

    Attributes
    ----------
    name : str
        Name of the loss function ('bce')
    """

    def __init__(self, from_logits: bool = False, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.from_logits = from_logits
        self.name = "bce"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Compute binary cross-entropy loss."""
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Clip predictions for numerical stability
        eps = 1e-7
        if self.from_logits:
            # Numerically stable BCE from logits
            # loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
            y_pred_clipped = np.clip(y_pred, -50, 50)
            loss = np.maximum(y_pred_clipped, 0) - y_pred_clipped * y_true
            loss += np.log1p(np.exp(-np.abs(y_pred_clipped)))
        else:
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            loss = -y_true * np.log(y_pred_clipped)
            loss -= (1 - y_true) * np.log(1 - y_pred_clipped)

        # Apply sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < loss.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            loss *= sample_weight

        if self.reduction == "none":
            return loss if loss.ndim == 1 else np.sum(loss, axis=-1)

        return self._apply_reduction(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of BCE loss."""
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        eps = 1e-7
        n = y_true.size

        if self.from_logits:
            # Gradient: sigmoid(y_pred) - y_true
            sigmoid = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -50, 50)))
            return (sigmoid - y_true) / n
        else:
            y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
            # Gradient: (y_pred - y_true) / (y_pred * (1 - y_pred))
            grad = (y_pred_clipped - y_true) / (y_pred_clipped * (1 - y_pred_clipped))
            return grad / n

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["from_logits"] = self.from_logits
        return config

    def __repr__(self) -> str:
        return f"BinaryCrossEntropy(from_logits={self.from_logits}, reduction='{self.reduction}')"


class CategoricalCrossEntropy(BaseLoss):
    """
    Categorical Cross-Entropy loss function - HYPER-OPTIMIZED.

    Computes the cross-entropy loss for multi-class classification.

    loss = -mean(sum(y_true * log(y_pred)))

    Parameters
    ----------
    from_logits : bool, default=False
        If True, y_pred is treated as logits and softmax is applied
        internally for numerical stability.
    reduction : str, default='mean'
        Type of reduction to apply

    Attributes
    ----------
    name : str
        Name of the loss function ('cce')
    """

    def __init__(self, from_logits: bool = False, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.from_logits = from_logits
        self.name = "cce"

    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Compute categorical cross-entropy loss."""
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        eps = 1e-7

        if self.from_logits:
            # Numerically stable softmax + log-loss
            # log_softmax = x - max(x) - log(sum(exp(x - max(x))))
            y_pred_max = np.max(y_pred, axis=-1, keepdims=True)
            y_pred_shifted = y_pred - y_pred_max
            log_softmax = y_pred_shifted - np.log(
                np.sum(np.exp(y_pred_shifted), axis=-1, keepdims=True) + eps
            )
            loss = -np.sum(y_true * log_softmax, axis=-1)
        else:
            y_pred_clipped = np.clip(y_pred, eps, 1.0)
            loss = -np.sum(y_true * np.log(y_pred_clipped), axis=-1)

        # Apply sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            loss *= sample_weight

        if self.reduction == "none":
            return loss

        return self._apply_reduction(loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute gradient of CCE loss."""
        y_true = np.ascontiguousarray(y_true, dtype=np.float64)
        y_pred = np.ascontiguousarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        n = y_true.shape[0]  # Batch size

        if self.from_logits:
            # Gradient: softmax(y_pred) - y_true
            y_pred_max = np.max(y_pred, axis=-1, keepdims=True)
            exp_pred = np.exp(y_pred - y_pred_max)
            softmax = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
            return (softmax - y_true) / n
        else:
            eps = 1e-7
            y_pred_clipped = np.clip(y_pred, eps, 1.0)
            # Gradient: -y_true / y_pred
            return -y_true / y_pred_clipped / n

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["from_logits"] = self.from_logits
        return config

    def __repr__(self) -> str:
        return f"CategoricalCrossEntropy(from_logits={self.from_logits}, reduction='{self.reduction}')"


# Aliases for convenience
MSE = MeanSquaredError
MAE = MeanAbsoluteError
BCE = BinaryCrossEntropy
CCE = CategoricalCrossEntropy
