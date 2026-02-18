"""
Dense (fully connected) layer implementation.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..initializers import BaseInitializer, Zeros, get_initializer
from ._base import BaseLayer


class Dense(BaseLayer):
    """
    Fully connected (dense) layer.

    Computes output = activation(input @ weights + bias)

    Parameters
    ----------
    units : int
        Number of neurons (output dimension)
    activation : str or callable, optional
        Activation function to use. Options: 'relu', 'sigmoid', 'tanh',
        'softmax', 'linear', None. Default is None (linear).
    use_bias : bool, default=True
        Whether to use a bias term
    kernel_initializer : str or Initializer, default='he_normal'
        Initializer for the weight matrix
    bias_initializer : str or Initializer, default='zeros'
        Initializer for the bias vector
    kernel_regularizer : callable, optional
        Regularizer function for the weights
    input_shape : tuple, optional
        Input shape for the layer (used for building the first layer)
    name : str, optional
        Layer name

    Attributes
    ----------
    weights : ndarray
        Weight matrix of shape (input_features, units)
    bias : ndarray
        Bias vector of shape (units,)
    grad_weights : ndarray
        Gradient of loss w.r.t. weights
    grad_bias : ndarray
        Gradient of loss w.r.t. bias

    Examples
    --------
    >>> # Create a dense layer with 128 neurons and ReLU activation
    >>> layer = Dense(128, activation='relu', input_shape=(784,))
    >>> layer.build((None, 784))
    >>> output = layer.forward(X)  # X.shape = (batch_size, 784)

    >>> # Use in Sequential model
    >>> model = Sequential([
    ...     Dense(128, activation='relu', input_shape=(784,)),
    ...     Dense(64, activation='relu'),
    ...     Dense(10, activation='softmax')
    ... ])
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, BaseInitializer] = "he_normal",
        bias_initializer: Union[str, BaseInitializer] = "zeros",
        kernel_regularizer: Optional[Callable] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")

        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self._input_shape_arg = input_shape

        # Parameters (initialized in build())
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        # Gradients (computed in backward())
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Cache for backward pass
        self._input_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None
        self._activation_cache: Optional[np.ndarray] = None

        # Store input shape if provided
        if input_shape is not None:
            self._input_shape = input_shape

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Initialize layer parameters based on input shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of the input data (batch_size, input_features)
        """
        if self.built:
            return

        if len(input_shape) < 2:
            raise ValueError(
                f"input_shape must have at least 2 dimensions, got {input_shape}"
            )

        input_dim = input_shape[-1]
        self._input_shape = input_shape

        # Initialize weights
        weight_initializer = get_initializer(self.kernel_initializer)
        self.weights = weight_initializer.initialize((input_dim, self.units))

        # Initialize bias
        if self.use_bias:
            bias_initializer = get_initializer(self.bias_initializer)
            if isinstance(bias_initializer, Zeros):
                self.bias = np.zeros(self.units, dtype=np.float64)
            else:
                self.bias = bias_initializer.initialize((self.units,))
        else:
            self.bias = None

        # Initialize gradient placeholders
        self.grad_weights = np.zeros_like(self.weights, dtype=np.float64)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias, dtype=np.float64)

        # Set output shape
        self._output_shape = (input_shape[0], self.units)

        self.built = True

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : ndarray of shape (batch_size, input_features)
            Input data
        training : bool
            Whether in training mode (unused for Dense, but required by interface)

        Returns
        -------
        output : ndarray of shape (batch_size, units)
            Layer output
        """
        # Ensure input is float64 for numerical stability
        x = np.asarray(x, dtype=np.float64)

        # Build layer if not already built
        if not self.built:
            self.build(x.shape)

        # Store input for backward pass
        self._input_cache = x

        # Linear transformation: z = x @ W + b
        self._z_cache = x @ self.weights
        if self.use_bias:
            self._z_cache = self._z_cache + self.bias

        # Apply activation
        self._activation_cache = self._apply_activation(self._z_cache)

        return self._activation_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass to compute gradients.

        Parameters
        ----------
        grad_output : ndarray of shape (batch_size, units)
            Gradient from the next layer (dL/d_output)

        Returns
        -------
        grad_input : ndarray of shape (batch_size, input_features)
            Gradient for the previous layer (dL/d_input)
        """
        if self._input_cache is None:
            raise RuntimeError("Forward pass must be called before backward pass")

        grad_output = np.asarray(grad_output, dtype=np.float64)
        batch_size = grad_output.shape[0]

        # Gradient through activation function
        grad_z = self._activation_gradient(grad_output, self._z_cache)

        # Compute gradients w.r.t. weights and bias
        # dL/dW = X.T @ grad_z / batch_size
        self.grad_weights = (self._input_cache.T @ grad_z) / batch_size

        # Add regularization gradient if present
        if self.kernel_regularizer is not None:
            reg_grad = self.kernel_regularizer.gradient(self.weights)
            self.grad_weights = self.grad_weights + reg_grad

        # dL/db = mean of grad_z over batch
        if self.use_bias:
            self.grad_bias = np.mean(grad_z, axis=0)

        # Compute gradient for previous layer
        # dL/dX = grad_z @ W.T
        grad_input = grad_z @ self.weights.T

        return grad_input

    def _apply_activation(self, z: np.ndarray) -> np.ndarray:
        """
        Apply activation function to the pre-activation values.

        Parameters
        ----------
        z : ndarray
            Pre-activation values (linear output)

        Returns
        -------
        output : ndarray
            Activated output
        """
        if self.activation is None or self.activation == "linear":
            return z

        if isinstance(self.activation, str):
            activation_name = self.activation.lower()

            if activation_name == "relu":
                return np.maximum(0, z)

            elif activation_name == "leaky_relu":
                alpha = 0.01
                return np.where(z > 0, z, alpha * z)

            elif activation_name == "sigmoid":
                # Numerically stable sigmoid
                z = np.clip(z, -500, 500)
                return 1.0 / (1.0 + np.exp(-z))

            elif activation_name == "tanh":
                return np.tanh(z)

            elif activation_name == "softmax":
                # Numerically stable softmax
                shifted = z - np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(shifted)
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)

            elif activation_name == "elu":
                alpha = 1.0
                return np.where(z > 0, z, alpha * (np.exp(z) - 1))

            elif activation_name == "selu":
                # SELU: scale * elu(x, alpha)
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                return scale * np.where(z > 0, z, alpha * (np.exp(z) - 1))

            elif activation_name == "swish" or activation_name == "silu":
                # Swish/SiLU: x * sigmoid(x)
                z_clipped = np.clip(z, -500, 500)
                return z * (1.0 / (1.0 + np.exp(-z_clipped)))

            elif activation_name == "gelu":
                # GELU approximation
                return (
                    0.5
                    * z
                    * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))
                )

            elif activation_name == "softplus":
                # Softplus: log(1 + exp(x))
                return np.log1p(np.exp(np.clip(z, -500, 500)))

            elif activation_name == "softsign":
                return z / (1.0 + np.abs(z))

            else:
                raise ValueError(f"Unknown activation function: '{self.activation}'")

        # Custom callable activation
        elif callable(self.activation):
            return self.activation(z)

        return z

    def _activation_gradient(
        self, grad_output: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient through activation function.

        Parameters
        ----------
        grad_output : ndarray
            Gradient from the next layer
        z : ndarray
            Pre-activation values (cached from forward pass)

        Returns
        -------
        grad_z : ndarray
            Gradient w.r.t. pre-activation values
        """
        if self.activation is None or self.activation == "linear":
            return grad_output

        if isinstance(self.activation, str):
            activation_name = self.activation.lower()

            if activation_name == "relu":
                return grad_output * (z > 0).astype(np.float64)

            elif activation_name == "leaky_relu":
                alpha = 0.01
                return grad_output * np.where(z > 0, 1.0, alpha)

            elif activation_name == "sigmoid":
                sig = self._apply_activation(z)
                return grad_output * sig * (1.0 - sig)

            elif activation_name == "tanh":
                tanh = np.tanh(z)
                return grad_output * (1.0 - tanh**2)

            elif activation_name == "softmax":
                # For softmax combined with cross-entropy loss,
                # the gradient is computed directly in the loss function.
                # When used standalone, we compute the Jacobian.
                # But typically this case is handled by the loss function.
                return grad_output

            elif activation_name == "elu":
                alpha = 1.0
                return grad_output * np.where(z > 0, 1.0, alpha * np.exp(z))

            elif activation_name == "selu":
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                return grad_output * scale * np.where(z > 0, 1.0, alpha * np.exp(z))

            elif activation_name == "swish" or activation_name == "silu":
                z_clipped = np.clip(z, -500, 500)
                sig = 1.0 / (1.0 + np.exp(-z_clipped))
                return grad_output * (sig + z * sig * (1.0 - sig))

            elif activation_name == "gelu":
                # GELU gradient (approximate)
                cdf = 0.5 * (
                    1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3))
                )
                pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
                return grad_output * (cdf + z * pdf)

            elif activation_name == "softplus":
                # Gradient of softplus is sigmoid
                z_clipped = np.clip(z, -500, 500)
                return grad_output * (1.0 / (1.0 + np.exp(-z_clipped)))

            elif activation_name == "softsign":
                return grad_output / (1.0 + np.abs(z)) ** 2

            else:
                return grad_output

        # Custom callable activation - assume it has backward method
        elif callable(self.activation) and hasattr(self.activation, "backward"):
            return self.activation.backward(grad_output, z)

        return grad_output

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Get layer parameters.

        Returns
        -------
        params : dict
            Dictionary with 'weights' and optionally 'bias'
        """
        params = {"weights": self.weights}
        if self.use_bias:
            params["bias"] = self.bias
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """
        Set layer parameters.

        Parameters
        ----------
        params : dict
            Dictionary with 'weights' and optionally 'bias'
        """
        if "weights" in params:
            self.weights = np.asarray(params["weights"], dtype=np.float64)
        if self.use_bias and "bias" in params:
            self.bias = np.asarray(params["bias"], dtype=np.float64)

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """
        Get parameter gradients.

        Returns
        -------
        gradients : dict
            Dictionary with 'weights' and optionally 'bias' gradients
        """
        grads = {"weights": self.grad_weights}
        if self.use_bias:
            grads["bias"] = self.grad_bias
        return grads

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns
        -------
        config : dict
            Layer configuration dictionary
        """
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation
                if isinstance(self.activation, str)
                else None,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer
                if isinstance(self.kernel_initializer, str)
                else "he_normal",
                "bias_initializer": self.bias_initializer
                if isinstance(self.bias_initializer, str)
                else "zeros",
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dense":
        """
        Create Dense layer from configuration.

        Parameters
        ----------
        config : dict
            Layer configuration dictionary

        Returns
        -------
        layer : Dense
            New Dense layer instance
        """
        return cls(**config)

    def __repr__(self) -> str:
        activation_str = f", activation='{self.activation}'" if self.activation else ""
        return f"Dense(units={self.units}{activation_str})"
