"""
Base class for neural network loss functions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseLoss(ABC):
    """
    Base class for all neural network loss functions in LynxLearn.

    All loss implementations (MSE, MAE, CrossEntropy, etc.) must inherit
    from this class and implement the abstract methods.

    Parameters
    ----------
    reduction : str, default='mean'
        Type of reduction to apply ('mean', 'sum', or 'none')

    Attributes
    ----------
    name : str
        Name of the loss function

    Examples
    --------
    >>> class CustomLoss(BaseLoss):
    ...     def compute(self, y_true, y_pred):
    ...         return np.mean((y_true - y_pred) ** 2)
    ...     def gradient(self, y_true, y_pred):
    ...         return 2 * (y_pred - y_true) / y_true.size
    """

    def __init__(self, reduction: str = "mean"):
        valid_reductions = ("mean", "sum", "none")
        if reduction not in valid_reductions:
            raise ValueError(
                f"reduction must be one of {valid_reductions}, got '{reduction}'"
            )
        self.reduction = reduction
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss value.

        Parameters
        ----------
        y_true : ndarray
            Ground truth values (true labels)
        y_pred : ndarray
            Predicted values (model output)

        Returns
        -------
        loss : float
            Computed loss value. Returns array if reduction='none'.
        """
        pass

    @abstractmethod
    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to predictions.

        Parameters
        ----------
        y_true : ndarray
            Ground truth values (true labels)
        y_pred : ndarray
            Predicted values (model output)

        Returns
        -------
        gradient : ndarray
            Gradient of loss with respect to y_pred (dL/dy_pred)
        """
        pass

    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, return_grad: bool = False
    ) -> Any:
        """
        Compute loss and optionally gradient.

        Parameters
        ----------
        y_true : ndarray
            Ground truth values
        y_pred : ndarray
            Predicted values
        return_grad : bool, default=False
            If True, also return gradient

        Returns
        -------
        loss : float
            Computed loss value
        gradient : ndarray, optional
            Only returned if return_grad=True
        """
        loss = self.compute(y_true, y_pred)
        if return_grad:
            return loss, self.gradient(y_true, y_pred)
        return loss

    def _apply_reduction(self, values: np.ndarray) -> Any:
        """
        Apply the specified reduction to the computed values.

        Parameters
        ----------
        values : ndarray
            Array of per-sample loss values

        Returns
        -------
        reduced : float or ndarray
            Reduced loss value (or unreduced if reduction='none')
        """
        if self.reduction == "mean":
            return np.mean(values)
        elif self.reduction == "sum":
            return np.sum(values)
        else:  # 'none'
            return values

    def get_config(self) -> Dict[str, Any]:
        """
        Get loss configuration for serialization.

        Returns
        -------
        config : dict
            Dictionary of loss configuration
        """
        return {
            "reduction": self.reduction,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseLoss":
        """
        Create loss from configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of loss configuration

        Returns
        -------
        loss : BaseLoss
            New loss instance
        """
        config = config.copy()
        config.pop("name", None)
        return cls(**config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduction='{self.reduction}')"
