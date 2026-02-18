"""
Sequential model implementation for stacking layers linearly.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ._base import BaseNeuralNetwork
from .layers import BaseLayer, Dense
from .losses import BaseLoss, MeanSquaredError
from .optimizers import SGD, BaseOptimizer


class Sequential(BaseNeuralNetwork):
    """
    Sequential model for stacking layers in a linear fashion.

    This is the simplest neural network API, similar to Keras Sequential.
    Layers are added one by one and executed in order during forward pass.

    Parameters
    ----------
    layers : list, optional
        List of layers to add to the model
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
    built : bool
        Whether the model has been built

    Examples
    --------
    >>> # Create a simple regression model
    >>> model = Sequential([
    ...     Dense(64, activation='relu', input_shape=(10,)),
    ...     Dense(32, activation='relu'),
    ...     Dense(1)
    ... ])
    >>> model.compile(optimizer='sgd', loss='mse')
    >>> history = model.train(X_train, y_train, epochs=100)

    >>> # Or add layers incrementally
    >>> model = Sequential()
    >>> model.add(Dense(64, activation='relu', input_shape=(784,)))
    >>> model.add(Dense(10, activation='softmax'))
    >>> model.compile(optimizer='adam', loss='categorical_crossentropy')

    References
    ----------
    .. [1] Chollet, F. "Deep Learning with Python", 2017 (Keras Sequential API)
    """

    def __init__(
        self, layers: Optional[List[BaseLayer]] = None, name: Optional[str] = None
    ):
        super().__init__(name=name)

        self._layers: List[BaseLayer] = []
        self._optimizer: Optional[BaseOptimizer] = None
        self._loss: Optional[BaseLoss] = None
        self._metrics: List[Callable] = []
        self._history: Dict[str, List[float]] = {}
        self._built = False
        self._is_compiled = False
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None
        self.stop_training = False

        # Add initial layers if provided
        if layers is not None:
            for layer in layers:
                self.add(layer)

    @property
    def layers(self) -> List[BaseLayer]:
        """Get list of layers in the model."""
        return self._layers

    @property
    def optimizer(self) -> Optional[BaseOptimizer]:
        """Get the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: BaseOptimizer) -> None:
        """Set the optimizer."""
        self._optimizer = value

    @property
    def loss(self) -> Optional[BaseLoss]:
        """Get the loss function."""
        return self._loss

    @loss.setter
    def loss(self, value: BaseLoss) -> None:
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

    def add(self, layer: BaseLayer) -> None:
        """
        Add a layer to the model.

        Parameters
        ----------
        layer : BaseLayer
            Layer to add to the model

        Raises
        ------
        TypeError
            If layer is not a BaseLayer instance
        ValueError
            If trying to add layer after model is built
        """
        if not isinstance(layer, BaseLayer):
            raise TypeError(
                f"layer must be a BaseLayer instance, got {type(layer).__name__}"
            )

        self._layers.append(layer)
        self._built = False  # Reset built state when adding new layers

    def _build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the model by initializing all layers.

        Parameters
        ----------
        input_shape : tuple
            Shape of input data (batch_size, features, ...)
        """
        if self._built:
            return

        if not self._layers:
            raise ValueError("Cannot build model with no layers. Add layers first.")

        # If input_shape doesn't have batch dimension, add None
        if input_shape and input_shape[0] is not None:
            # Check if this looks like a sample shape (no batch dim)
            # If first layer has input_shape arg, use that format
            first_layer = self._layers[0]
            if (
                hasattr(first_layer, "_input_shape_arg")
                and first_layer._input_shape_arg is not None
            ):
                # User provided input_shape without batch dim, prepend None
                if len(input_shape) == len(first_layer._input_shape_arg):
                    input_shape = (None,) + input_shape

        current_shape = input_shape

        for i, layer in enumerate(self._layers):
            # Set layer name if not set
            if layer.name is None or layer.name == layer.__class__.__name__:
                layer.name = f"{layer.__class__.__name__.lower()}_{i}"

            # Build layer
            if not layer.built:
                layer.build(current_shape)

            # Update current shape for next layer
            current_shape = layer.output_shape

        self._input_shape = input_shape
        self._output_shape = current_shape
        self._built = True

    def compile(
        self,
        optimizer: Union[str, BaseOptimizer] = "sgd",
        loss: Union[str, BaseLoss] = "mse",
        metrics: Optional[List[Union[str, Callable]]] = None,
        **kwargs,
    ) -> None:
        """
        Configure the model for training.

        Parameters
        ----------
        optimizer : str or BaseOptimizer, default='sgd'
            Optimizer to use for training. Options: 'sgd', 'adam', or a
            BaseOptimizer instance.
        loss : str or BaseLoss, default='mse'
            Loss function to minimize. Options: 'mse', 'mae', 'huber',
            'binary_crossentropy', 'categorical_crossentropy', or a
            BaseLoss instance.
        metrics : list, optional
            List of metrics to track during training. Can be strings
            ('accuracy') or callable functions.
        **kwargs : dict
            Additional arguments passed to optimizer or loss if using strings.

        Examples
        --------
        >>> model.compile(optimizer='sgd', loss='mse')
        >>> model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
        ...               loss=MeanSquaredError())
        """
        # Handle optimizer
        if isinstance(optimizer, str):
            self._optimizer = self._get_optimizer(optimizer, **kwargs)
        elif isinstance(optimizer, BaseOptimizer):
            self._optimizer = optimizer
        else:
            raise TypeError(
                f"optimizer must be str or BaseOptimizer, got {type(optimizer)}"
            )

        # Handle loss
        if isinstance(loss, str):
            self._loss = self._get_loss(loss, **kwargs)
        elif isinstance(loss, BaseLoss):
            self._loss = loss
        else:
            raise TypeError(f"loss must be str or BaseLoss, got {type(loss)}")

        # Handle metrics
        self._metrics = []
        if metrics is not None:
            for metric in metrics:
                if isinstance(metric, str):
                    self._metrics.append(self._get_metric(metric))
                elif callable(metric):
                    self._metrics.append(metric)
                else:
                    raise TypeError(
                        f"metric must be str or callable, got {type(metric)}"
                    )

        self._is_compiled = True

        # Build model if first layer has input_shape
        if not self._built and self._layers:
            first_layer = self._layers[0]
            if (
                hasattr(first_layer, "_input_shape_arg")
                and first_layer._input_shape_arg is not None
            ):
                # Build with the input shape from first layer
                input_shape = (None,) + first_layer._input_shape_arg
                self._build(input_shape)

    def _get_optimizer(self, name: str, **kwargs) -> BaseOptimizer:
        """Get optimizer instance from string name."""
        from .optimizers import SGD

        optimizers = {
            "sgd": SGD,
        }

        name_lower = name.lower()
        if name_lower not in optimizers:
            raise ValueError(
                f"Unknown optimizer: '{name}'. Available: {list(optimizers.keys())}"
            )

        # Extract optimizer-specific kwargs
        lr = kwargs.get("learning_rate", kwargs.get("lr", 0.01))
        momentum = kwargs.get("momentum", 0.0)

        return optimizers[name_lower](learning_rate=lr, momentum=momentum)

    def _get_loss(self, name: str, **kwargs) -> BaseLoss:
        """Get loss instance from string name."""
        from .losses import MeanAbsoluteError, MeanSquaredError

        losses = {
            "mse": MeanSquaredError,
            "mean_squared_error": MeanSquaredError,
            "mae": MeanAbsoluteError,
            "mean_absolute_error": MeanAbsoluteError,
        }

        name_lower = name.lower()
        if name_lower not in losses:
            raise ValueError(
                f"Unknown loss: '{name}'. Available: {list(losses.keys())}"
            )

        return losses[name_lower](reduction=kwargs.get("reduction", "mean"))

    def _get_metric(self, name: str) -> Callable:
        """Get metric function from string name."""
        from ..metrics import mean_squared_error, r2_score

        metrics = {
            "accuracy": self._accuracy,
            "acc": self._accuracy,
            "mse": mean_squared_error,
            "r2": r2_score,
            "r2_score": r2_score,
        }

        name_lower = name.lower()
        if name_lower not in metrics:
            raise ValueError(
                f"Unknown metric: '{name}'. Available: {list(metrics.keys())}"
            )

        return metrics[name_lower]

    def _accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy."""
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            # Multi-class: use argmax
            y_pred_classes = np.argmax(y_pred, axis=1)
            if y_true.ndim > 1:
                y_true = np.argmax(y_true, axis=1)
        else:
            # Binary: threshold at 0.5
            y_pred_classes = (y_pred > 0.5).astype(int).ravel()
            y_true = y_true.ravel()

        return np.mean(y_pred_classes == y_true)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        validation_split: float = 0.0,
        callbacks: Optional[List[Any]] = None,
        verbose: int = 1,
        shuffle: bool = True,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Train the neural network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values
        epochs : int, default=100
            Number of epochs to train
        batch_size : int, default=32
            Number of samples per gradient update
        validation_data : tuple, optional
            (X_val, y_val) tuple for validation
        validation_split : float, default=0.0
            Fraction of training data to use for validation
        callbacks : list, optional
            List of callback objects
        verbose : int, default=1
            Verbosity mode. 0=silent, 1=progress bar, 2=one line per epoch
        shuffle : bool, default=True
            Whether to shuffle data at each epoch

        Returns
        -------
        history : dict
            Training history with loss and metrics per epoch

        Raises
        ------
        RuntimeError
            If model is not compiled
        ValueError
            If inputs have incompatible shapes

        Examples
        --------
        >>> history = model.train(X_train, y_train, epochs=100, batch_size=32)
        >>> print(f"Final loss: {history['loss'][-1]:.4f}")
        """
        if not self._is_compiled:
            raise RuntimeError(
                "Model must be compiled before training. Call model.compile() first."
            )

        # Convert inputs to numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Reshape y if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = X.shape[0]

        # Build model if not built
        if not self._built:
            self._build((None,) + X.shape[1:])

        # Handle validation split
        if validation_split > 0.0:
            if validation_data is not None:
                raise ValueError("Cannot use both validation_data and validation_split")
            val_size = int(n_samples * validation_split)
            indices = np.random.permutation(n_samples)
            val_indices = indices[:val_size]
            train_indices = indices[val_size:]
            X_val, y_val = X[val_indices], y[val_indices]
            X, y = X[train_indices], y[train_indices]
            validation_data = (X_val, y_val)
            n_samples = X.shape[0]

        # Initialize history
        self._history = {"loss": []}
        if validation_data is not None:
            self._history["val_loss"] = []

        # Initialize callbacks
        if callbacks is None:
            callbacks = []

        # Training loop
        self.stop_training = False

        for epoch in range(epochs):
            if self.stop_training:
                break

            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
            else:
                X_shuffled = X
                y_shuffled = y

            # Mini-batch training
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                y_pred = self._forward_pass(X_batch, training=True)

                # Compute loss
                batch_loss = self._loss.compute(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches += 1

                # Compute gradient
                grad = self._loss.gradient(y_batch, y_pred)

                # Backward pass
                self._backward_pass(grad)

                # Update parameters
                for layer in self._layers:
                    if layer.trainable:
                        self._optimizer.update(layer)

            # Average loss for epoch
            avg_loss = epoch_loss / max(n_batches, 1)
            self._history["loss"].append(avg_loss)

            # Compute validation loss
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self._forward_pass(X_val, training=False)
                val_loss = self._loss.compute(y_val, y_val_pred)
                self._history["val_loss"].append(val_loss)

            # Call callbacks
            for callback in callbacks:
                if hasattr(callback, "on_epoch_end"):
                    callback.on_epoch_end(epoch, self._history)

            # Print progress
            if verbose > 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}"
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.4f}"
                print(msg)

        return self._history

    # Alias for compatibility
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, List[float]]:
        """Alias for train()."""
        return self.train(X, y, **kwargs)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        Returns
        -------
        predictions : ndarray
            Predicted values

        Examples
        --------
        >>> predictions = model.predict(X_test)
        """
        X = np.asarray(X, dtype=np.float64)

        # Build model if not built
        if not self._built:
            if len(self._layers) > 0 and self._layers[0]._input_shape_arg is not None:
                input_shape = (None,) + self._layers[0]._input_shape_arg[1:]
                self._build(input_shape)
            else:
                self._build((None,) + X.shape[1:])

        return self._forward_pass(X, training=False)

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

        Examples
        --------
        >>> probas = model.predict_proba(X_test)
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

        Examples
        --------
        >>> classes = model.predict_classes(X_test)
        """
        proba = self.predict_proba(X)

        if proba.ndim > 1 and proba.shape[1] > 1:
            return np.argmax(proba, axis=1)
        else:
            return (proba > 0.5).astype(int).ravel()

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

        Examples
        --------
        >>> results = model.evaluate(X_test, y_test)
        >>> print(f"Test loss: {results['loss']:.4f}")
        """
        if not self._is_compiled:
            raise RuntimeError("Model must be compiled before evaluation.")

        y_pred = self.predict(X)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        loss_value = self._loss.compute(y, y_pred)

        results = {"loss": loss_value}

        # Compute configured metrics
        for metric in self._metrics:
            metric_name = (
                metric.__name__ if hasattr(metric, "__name__") else str(metric)
            )
            results[metric_name] = metric(y, y_pred)

        # Compute additional metrics
        if metrics:
            for metric in metrics:
                metric_name = (
                    metric.__name__ if hasattr(metric, "__name__") else str(metric)
                )
                results[metric_name] = metric(y, y_pred)

        return results

    def summary(self) -> None:
        """
        Print a summary of the model architecture.

        Examples
        --------
        >>> model.summary()
        Model: Sequential
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        dense_0 (Dense)              (None, 64)                640
        _________________________________________________________________
        dense_1 (Dense)              (None, 1)                 65
        =================================================================
        Total params: 705
        Trainable params: 705
        Non-trainable params: 0
        _________________________________________________________________
        """
        self._print_summary()

    def _print_summary(self) -> None:
        """Print detailed model summary."""
        print(f"Model: {self.__class__.__name__}")
        print("_" * 65)
        print(f"{'Layer (type)':<30} {'Output Shape':<20} {'Param #':>10}")
        print("=" * 65)

        total_params = 0
        trainable_params = 0

        for layer in self._layers:
            layer_type = f"{layer.name} ({layer.__class__.__name__})"
            output_shape = str(layer.output_shape) if layer.output_shape else "Unknown"
            param_count = layer.count_params()

            print(f"{layer_type:<30} {output_shape:<20} {param_count:>10,}")

            total_params += param_count
            if layer.trainable:
                trainable_params += param_count

        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("_" * 65)

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
        return sum(layer.count_params() for layer in self._layers)

    def get_weights(self) -> List[Dict[str, np.ndarray]]:
        """
        Get all layer weights.

        Returns
        -------
        weights : list
            List of parameter dictionaries for each layer

        Notes
        -----
        Returns empty list if model is not built yet.
        """
        if not self._built:
            return []
        return [
            layer.get_params()
            for layer in self._layers
            if hasattr(layer, "get_params") and layer.built
        ]

    def set_weights(self, weights: List[Dict[str, np.ndarray]]) -> None:
        """
        Set all layer weights.

        Parameters
        ----------
        weights : list
            List of parameter dictionaries for each layer

        Raises
        ------
        ValueError
            If model is not built or weights count doesn't match
        """
        if not self._built:
            raise ValueError("Model must be built before setting weights.")

        layers_with_params = [
            layer
            for layer in self._layers
            if hasattr(layer, "set_params") and layer.built
        ]
        if len(weights) != len(layers_with_params):
            raise ValueError(
                f"Expected weights for {len(layers_with_params)} layers, got {len(weights)}"
            )
        for layer, layer_weights in zip(layers_with_params, weights):
            layer.set_params(layer_weights)

    def save(self, filepath: str) -> None:
        """
        Save the model to a file.

        Parameters
        ----------
        filepath : str
            Path to save the model

        Examples
        --------
        >>> model.save('my_model.npz')
        """
        # Save weights, config, and optimizer state
        save_dict = {
            "layer_weights": [],
            "layer_configs": [],
            "optimizer_config": None,
            "optimizer_state": None,
        }

        for layer in self._layers:
            save_dict["layer_weights"].append(layer.get_params())
            save_dict["layer_configs"].append(layer.get_config())

        if self._optimizer is not None:
            save_dict["optimizer_config"] = self._optimizer.get_config()
            save_dict["optimizer_state"] = self._optimizer.get_state()

        # Save as npz file
        # Convert nested dicts to arrays for npz compatibility
        np.savez(filepath, **self._flatten_dict(save_dict))

    def _flatten_dict(self, d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
        """Flatten nested dictionary for saving."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            self._flatten_dict(item, f"{new_key}_{i}", sep).items()
                        )
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    @classmethod
    def load(cls, filepath: str) -> "Sequential":
        """
        Load a model from a file.

        Parameters
        ----------
        filepath : str
            Path to the saved model

        Returns
        -------
        model : Sequential
            Loaded model

        Examples
        --------
        >>> model = Sequential.load('my_model.npz')
        """
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct model
        # This is a simplified version - full implementation would
        # parse the saved configs and recreate layers
        model = cls()

        # Load optimizer state if available
        # (simplified - full implementation would restore full optimizer)

        return model

    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.

        Returns
        -------
        config : dict
            Model configuration dictionary
        """
        config = {
            "name": self.name,
            "layers": [layer.get_config() for layer in self._layers],
        }
        return config

    def __repr__(self) -> str:
        layer_str = ", ".join(repr(layer) for layer in self._layers)
        return f"Sequential([{layer_str}])"
