"""
Base class for neural network layers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseLayer(ABC):
    """
    Base class for all neural network layers in LynxLearn.

    All layer implementations (Dense, Conv2D, LSTM, etc.) must inherit
    from this class and implement the abstract methods.

    Parameters
    ----------
    name : str, optional
        Name of the layer

    Attributes
    ----------
    built : bool
        Whether the layer has been built (parameters initialized)
    trainable : bool
        Whether the layer's parameters can be updated during training
    training : bool
        Whether the layer is in training mode

    Examples
    --------
    >>> class CustomLayer(BaseLayer):
    ...     def build(self, input_shape):
    ...         self.weights = np.random.randn(*input_shape)
    ...         self.built = True
    ...     def forward(self, x, training=True):
    ...         return x @ self.weights
    ...     def backward(self, grad_output):
    ...         return grad_output @ self.weights.T
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.built = False
        self.trainable = True
        self.training = True
        self._input_shape = None
        self._output_shape = None

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Initialize layer parameters based on input shape.

        This method is called once to create the layer's weights and biases.
        It should set `self.built = True` after initialization.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, features, ...)
        """
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : ndarray
            Input data
        training : bool
            Whether in training mode (affects Dropout, BatchNorm, etc.)

        Returns
        -------
        output : ndarray
            Layer output
        """
        pass

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass to compute gradients.

        Parameters
        ----------
        grad_output : ndarray
            Gradient from the next layer (dL/d_output)

        Returns
        -------
        grad_input : ndarray
            Gradient for the previous layer (dL/d_input)
        """
        pass

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters.

        Returns
        -------
        params : dict
            Dictionary of parameter names to arrays
        """
        return {}

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set layer parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names to arrays
        """
        pass

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """
        Get parameter gradients computed during backward pass.

        Returns
        -------
        gradients : dict
            Dictionary of parameter names to gradient arrays
        """
        return {}

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns
        -------
        config : dict
            Layer configuration dictionary
        """
        return {
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseLayer":
        """
        Create layer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Layer configuration dictionary

        Returns
        -------
        layer : BaseLayer
            New layer instance
        """
        return cls(**config)

    def set_training(self, mode: bool = True) -> None:
        """
        Set the training mode for the layer.

        Parameters
        ----------
        mode : bool
            True for training mode, False for inference mode
        """
        self.training = mode

    def count_params(self) -> int:
        """
        Count the total number of trainable parameters.

        Returns
        -------
        count : int
            Total number of trainable parameters
        """
        if not self.built:
            return 0

        total = 0
        for param in self.get_params().values():
            total += param.size
        return total

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """Get input shape."""
        return self._input_shape

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Get output shape."""
        return self._output_shape

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Call forward pass."""
        return self.forward(x, training=training)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
