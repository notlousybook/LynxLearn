"""
Base class for neural network optimizers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseOptimizer(ABC):
    """
    Base class for all neural network optimizers in LynxLearn.

    All optimizer implementations (SGD, Adam, RMSprop, etc.) must inherit
    from this class and implement the abstract methods.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for parameter updates

    Attributes
    ----------
    learning_rate : float
        Current learning rate
    iterations : int
        Number of update iterations performed

    Examples
    --------
    >>> class CustomOptimizer(BaseOptimizer):
    ...     def update(self, layer):
    ...         params = layer.get_params()
    ...         grads = layer.get_gradients()
    ...         for key in params:
    ...             params[key] -= self.learning_rate * grads[key]
    ...         layer.set_params(params)
    """

    def __init__(self, learning_rate: float = 0.01):
        self._learning_rate = learning_rate
        self.iterations = 0

    @property
    def learning_rate(self) -> float:
        """Get the current learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set the learning rate."""
        if value <= 0:
            raise ValueError(f"Learning rate must be positive, got {value}")
        self._learning_rate = value

    @abstractmethod
    def update(self, layer: Any) -> None:
        """
        Update layer parameters using computed gradients.

        This method should:
        1. Get parameters and gradients from the layer
        2. Apply the optimization algorithm
        3. Set the updated parameters back on the layer

        Parameters
        ----------
        layer : BaseLayer
            Layer with parameters to update. Must implement
            get_params(), set_params(), and get_gradients() methods.
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get optimizer state for serialization.

        Returns
        -------
        state : dict
            Dictionary containing optimizer state variables
        """
        return {
            "learning_rate": self._learning_rate,
            "iterations": self.iterations,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore optimizer state from dictionary.

        Parameters
        ----------
        state : dict
            Dictionary containing optimizer state variables
        """
        self._learning_rate = state.get("learning_rate", self._learning_rate)
        self.iterations = state.get("iterations", 0)

    def reset(self) -> None:
        """
        Reset optimizer state.

        Called when starting training from scratch.
        """
        self.iterations = 0

    def increment_iterations(self) -> None:
        """Increment the iteration counter."""
        self.iterations += 1

    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration for serialization.

        Returns
        -------
        config : dict
            Dictionary of optimizer configuration
        """
        return {
            "learning_rate": self._learning_rate,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseOptimizer":
        """
        Create optimizer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of optimizer configuration

        Returns
        -------
        optimizer : BaseOptimizer
            New optimizer instance
        """
        return cls(**config)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(learning_rate={self._learning_rate})"
