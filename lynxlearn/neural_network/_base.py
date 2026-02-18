"""
Base class for neural network models.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class BaseNeuralNetwork(ABC):
    """
    Base class for all neural network models in LynxLearn.

    All neural network implementations (Sequential, Functional API, etc.)
    must inherit from this class and implement the abstract methods.

    Parameters
    ----------
    name : str, optional
        Name of the model

    Attributes
    ----------
    layers : list
        List of layers in the model
    optimizer : BaseOptimizer
        Optimizer for training
    loss : BaseLoss
        Loss function
    history : dict
        Training history with loss and metrics per epoch
    stop_training : bool
        Flag to stop training (used by callbacks)

    Examples
    --------
    >>> class CustomModel(BaseNeuralNetwork):
    ...     def compile(self, optimizer, loss, metrics=None, **kwargs):
    ...         self.optimizer = optimizer
    ...         self.loss = loss
    ...     def train(self, X, y, **kwargs):
    ...         # Training implementation
    ...         pass
    ...     def predict(self, X):
    ...         # Prediction implementation
    ...         pass
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self._layers: List[Any] = []
        self._optimizer: Optional[Any] = None
        self._loss: Optional[Any] = None
        self._metrics: List[Callable] = []
        self._history: Dict[str, List[float]] = {}
        self._is_compiled = False
        self._built = False
        self.stop_training = False

    @property
    def layers(self) -> List[Any]:
        """Get list of layers in the model."""
        return self._layers

    @property
    def optimizer(self) -> Optional[Any]:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Any) -> None:
        """Set the optimizer."""
        self._optimizer = value

    @property
    def loss(self) -> Optional[Any]:
        """Get the loss function."""
        return self._loss

    @loss.setter
    def loss(self, value: Any) -> None:
        """Set the loss function."""
        self._loss = value

    @property
    def history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self._history

    @property
    def built(self) -> bool:
        """Check if model is built."""
        return self._built

    @abstractmethod
    def compile(
        self,
        optimizer: Union[str, Any],
        loss: Union[str, Any],
        metrics: Optional[List[Union[str, Callable]]] = None,
        **kwargs,
    ) -> None:
        """
        Configure the model for training.

        Parameters
        ----------
        optimizer : str or BaseOptimizer
            Optimizer to use for training
        loss : str or BaseLoss
            Loss function to minimize
        metrics : list, optional
            List of metrics to track during training
        **kwargs : dict
            Additional arguments
        """
        pass

    @abstractmethod
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Target values
        epochs : int
            Number of epochs to train
        batch_size : int
            Number of samples per gradient update
        **kwargs : dict
            Additional training arguments

        Returns
        -------
        history : dict
            Training history with loss and metrics per epoch
        """
        pass

    # Alias for compatibility
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, List[float]]:
        """
        Alias for train().

        Parameters
        ----------
        X : ndarray
            Training data
        y : ndarray
            Target values
        **kwargs : dict
            Additional training arguments

        Returns
        -------
        history : dict
            Training history
        """
        return self.train(X, y, **kwargs)

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : ndarray
            Input data

        Returns
        -------
        predictions : ndarray
            Predicted values
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (for classification).

        Parameters
        ----------
        X : ndarray
            Input data

        Returns
        -------
        probabilities : ndarray
            Class probabilities
        """
        return self.predict(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (for classification).

        Parameters
        ----------
        X : ndarray
            Input data

        Returns
        -------
        classes : ndarray
            Predicted class labels
        """
        proba = self.predict_proba(X)

        if proba.ndim > 1 and proba.shape[1] > 1:
            return np.argmax(proba, axis=1)
        else:
            return (proba > 0.5).astype(int).ravel()

    @abstractmethod
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[List[Callable]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Parameters
        ----------
        X : ndarray
            Test data
        y : ndarray
            True labels
        metrics : list, optional
            Additional metrics to compute

        Returns
        -------
        results : dict
            Dictionary with loss and metric values
        """
        pass

    @abstractmethod
    def summary(self) -> None:
        """
        Print a summary of the model architecture.
        """
        pass

    def _forward_pass(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Execute forward pass through all layers.

        Parameters
        ----------
        X : ndarray
            Input data
        training : bool
            Whether in training mode

        Returns
        -------
        output : ndarray
            Model output
        """
        output = X
        for layer in self._layers:
            output = layer.forward(output, training=training)
        return output

    def _backward_pass(self, grad: np.ndarray) -> np.ndarray:
        """
        Execute backward pass through all layers.

        Parameters
        ----------
        grad : ndarray
            Gradient from loss function

        Returns
        -------
        grad : ndarray
            Gradient after backpropagation
        """
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def count_params(self) -> int:
        """
        Count the total number of parameters in the model.

        Returns
        -------
        count : int
            Total number of parameters
        """
        if not self._built:
            return 0
        return sum(
            layer.count_params()
            for layer in self._layers
            if hasattr(layer, "count_params")
        )

    def get_weights(self) -> List[Dict[str, np.ndarray]]:
        """
        Get all layer weights.

        Returns
        -------
        weights : list
            List of parameter dictionaries for each layer
        """
        return [
            layer.get_params() for layer in self._layers if hasattr(layer, "get_params")
        ]

    def set_weights(self, weights: List[Dict[str, np.ndarray]]) -> None:
        """
        Set all layer weights.

        Parameters
        ----------
        weights : list
            List of parameter dictionaries for each layer
        """
        layers_with_params = [
            layer for layer in self._layers if hasattr(layer, "set_params")
        ]
        if len(weights) != len(layers_with_params):
            raise ValueError(
                f"Expected weights for {len(layers_with_params)} layers, got {len(weights)}"
            )
        for layer, layer_weights in zip(layers_with_params, weights):
            layer.set_params(layer_weights)

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.

        Returns
        -------
        config : dict
            Model configuration dictionary
        """
        return {
            "name": self.name,
            "layers": [
                layer.get_config()
                for layer in self._layers
                if hasattr(layer, "get_config")
            ],
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
