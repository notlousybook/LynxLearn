"""
Mean Squared Error (MSE) loss function implementation.
"""

from typing import Any, Dict, Optional, Union

import numpy as np

from ._base import BaseLoss


class MeanSquaredError(BaseLoss):
    """
    Mean Squared Error loss function.

    Computes the mean of squared differences between true and predicted values.

    loss = mean((y_true - y_pred)^2)

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

        Raises
        ------
        ValueError
            If y_true and y_pred have incompatible shapes.
        """
        # Convert to numpy arrays with float64 for numerical stability
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Compute squared error
        error = y_true - y_pred
        squared_error = np.square(error)

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            # Broadcast weights to match squared_error shape
            if sample_weight.ndim < squared_error.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            squared_error = squared_error * sample_weight

        # Sum over output dimensions (keep per-sample losses)
        if squared_error.ndim > 1:
            per_sample_loss = np.sum(
                squared_error, axis=tuple(range(1, squared_error.ndim))
            )
        else:
            per_sample_loss = squared_error

        # Apply reduction
        if self.reduction == "none":
            return per_sample_loss

        if sample_weight is not None:
            # Weighted mean
            if self.reduction == "mean":
                return np.sum(per_sample_loss) / np.sum(sample_weight)
            else:  # sum
                return np.sum(per_sample_loss)
        else:
            return self._apply_reduction(per_sample_loss)

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MSE loss with respect to predictions.

        The gradient is computed as:
            d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n

        For batch training, this returns the gradient averaged over samples.

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

        Notes
        -----
        The gradient formula for MSE is derived as follows:
            L = (y_true - y_pred)^2
            dL/d(y_pred) = 2 * (y_pred - y_true)

        For numerical stability and proper gradient scaling, we average
        over the total number of elements.
        """
        # Convert to numpy arrays with float64 for numerical stability
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Ensure shapes match
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Number of elements (for averaging)
        n = y_true.size

        # Gradient: d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n
        gradient = 2.0 * (y_pred - y_true) / n

        return gradient

    def get_config(self) -> Dict[str, Any]:
        """
        Get loss configuration for serialization.

        Returns
        -------
        config : dict
            Dictionary of loss configuration
        """
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MeanSquaredError":
        """
        Create MSE loss from configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of loss configuration

        Returns
        -------
        loss : MeanSquaredError
            New MeanSquaredError instance
        """
        config = config.copy()
        config.pop("name", None)
        return cls(**config)

    def __repr__(self) -> str:
        return f"MeanSquaredError(reduction='{self.reduction}')"


class MeanAbsoluteError(BaseLoss):
    """
    Mean Absolute Error (MAE) loss function.

    Computes the mean of absolute differences between true and predicted values.

    loss = mean(|y_true - y_pred|)

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
        # Convert to numpy arrays
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        # Compute absolute error
        error = np.abs(y_true - y_pred)

        # Apply sample weights if provided
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < error.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            error = error * sample_weight

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
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        n = y_true.size
        gradient = np.sign(y_pred - y_true) / n

        return gradient

    def __repr__(self) -> str:
        return f"MeanAbsoluteError(reduction='{self.reduction}')"


class HuberLoss(BaseLoss):
    """
    Huber loss function.

    Combines MSE for small errors and MAE for large errors,
    making it less sensitive to outliers than MSE.

    loss = { 0.5 * (y_true - y_pred)^2  if |y_true - y_pred| <= delta
           { delta * |y_true - y_pred| - 0.5 * delta^2  otherwise

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
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        error = y_true - y_pred
        abs_error = np.abs(error)

        # Quadratic for small errors, linear for large errors
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic

        loss = 0.5 * quadratic**2 + self.delta * linear

        # Apply sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.ndim < loss.ndim:
                sample_weight = np.expand_dims(sample_weight, axis=-1)
            loss = loss * sample_weight

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
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true.shape={y_true.shape}, "
                f"y_pred.shape={y_pred.shape}"
            )

        error = y_pred - y_true
        abs_error = np.abs(error)

        n = y_true.size

        # Gradient: -error for small errors, -delta * sign(error) for large errors
        gradient = np.where(
            abs_error <= self.delta, error / n, self.delta * np.sign(error) / n
        )

        return gradient

    def get_config(self) -> Dict[str, Any]:
        """Get loss configuration."""
        config = super().get_config()
        config["delta"] = self.delta
        return config

    def __repr__(self) -> str:
        return f"HuberLoss(delta={self.delta}, reduction='{self.reduction}')"


# Aliases for convenience
MSE = MeanSquaredError
MAE = MeanAbsoluteError
