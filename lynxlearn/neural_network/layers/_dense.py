"""
Universal Dense (Fully Connected) Layer.

A highly customizable dense layer supporting:
- All numeric dtypes: float16, float32, float64, bfloat16
- Custom weight/bias initializers
- String or callable activations
- Weight regularization (L1, L2)
- Activity regularization
- Weight constraints
- Mixed precision training
- Quantization-aware training hooks
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..initializers import (
    BaseInitializer,
    Constant,
    HeNormal,
    HeUniform,
    LeCunNormal,
    XavierNormal,
    XavierUniform,
    Zeros,
    get_initializer,
)
from ._base import BaseLayer

# Try to import bfloat16 support
try:
    from ml_dtypes import bfloat16

    BF16_AVAILABLE = True
except ImportError:
    BF16_AVAILABLE = False
    bfloat16 = None

# Supported dtypes
DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "single": np.float32,
    "double": np.float64,
    "half": np.float16,
}

if BF16_AVAILABLE:
    DTYPE_MAP["bfloat16"] = bfloat16
    DTYPE_MAP["bf16"] = bfloat16


# =============================================================================
# Activation Functions Registry
# =============================================================================


class ActivationRegistry:
    """
    Registry for activation functions with their forward and backward passes.

    Supports both string identifiers and custom callables.
    """

    _registry: Dict[str, Dict[str, Callable]] = {}

    @classmethod
    def register(cls, name: str, forward: Callable, backward: Callable) -> None:
        """Register a new activation function."""
        cls._registry[name.lower()] = {"forward": forward, "backward": backward}

    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Callable]]:
        """Get activation function by name."""
        return cls._registry.get(name.lower())

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available activations."""
        return list(cls._registry.keys())


# Built-in activation functions
def _relu_forward(z: np.ndarray) -> np.ndarray:
    """ReLU forward pass."""
    return np.maximum(z, 0)


def _relu_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """ReLU backward pass."""
    return grad * (z > 0)


def _leaky_relu_forward(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU forward pass."""
    return np.where(z > 0, z, alpha * z)


def _leaky_relu_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray, alpha: float = 0.01
) -> np.ndarray:
    """Leaky ReLU backward pass."""
    return grad * np.where(z > 0, 1.0, alpha)


def _sigmoid_forward(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid forward pass."""
    out = np.empty_like(z)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    out[neg_mask] = exp_z / (1.0 + exp_z)
    return out


def _sigmoid_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray
) -> np.ndarray:
    """Sigmoid backward pass."""
    return grad * output * (1.0 - output)


def _tanh_forward(z: np.ndarray) -> np.ndarray:
    """Tanh forward pass."""
    return np.tanh(z)


def _tanh_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Tanh backward pass."""
    return grad * (1.0 - output * output)


def _softmax_forward(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax forward pass."""
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _softmax_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray
) -> np.ndarray:
    """Softmax backward pass (for use with cross-entropy)."""
    return grad


def _elu_forward(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU forward pass."""
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))


def _elu_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """ELU backward pass."""
    return grad * np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))


def _selu_forward(z: np.ndarray) -> np.ndarray:
    """SELU forward pass."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))


def _selu_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """SELU backward pass."""
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return grad * scale * np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))


def _swish_forward(z: np.ndarray) -> np.ndarray:
    """Swish/SiLU forward pass."""
    sig = _sigmoid_forward(z)
    return z * sig


def _swish_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Swish backward pass."""
    sig = _sigmoid_forward(z)
    return grad * (sig + z * sig * (1.0 - sig))


def _gelu_forward(z: np.ndarray) -> np.ndarray:
    """GELU forward pass (approximate)."""
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))


def _gelu_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """GELU backward pass (approximate)."""
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z**3)))
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return grad * (cdf + z * pdf)


def _softplus_forward(z: np.ndarray) -> np.ndarray:
    """Softplus forward pass."""
    return np.log1p(np.exp(np.clip(z, -500, 500)))


def _softplus_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray
) -> np.ndarray:
    """Softplus backward pass (gradient is sigmoid)."""
    return grad * _sigmoid_forward(z)


def _softsign_forward(z: np.ndarray) -> np.ndarray:
    """Softsign forward pass."""
    return z / (1.0 + np.abs(z))


def _softsign_backward(
    grad: np.ndarray, z: np.ndarray, output: np.ndarray
) -> np.ndarray:
    """Softsign backward pass."""
    return grad / (1.0 + np.abs(z)) ** 2


def _linear_forward(z: np.ndarray) -> np.ndarray:
    """Linear/identity forward pass."""
    return z


def _linear_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Linear backward pass."""
    return grad


def _mish_forward(z: np.ndarray) -> np.ndarray:
    """Mish forward pass."""
    return z * np.tanh(_softplus_forward(z))


def _mish_backward(grad: np.ndarray, z: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Mish backward pass."""
    sp = _softplus_forward(z)
    tanh_sp = np.tanh(sp)
    sig = _sigmoid_forward(z)
    return grad * (tanh_sp + z * (1 - tanh_sp**2) * sig)


# Register built-in activations
ActivationRegistry.register("relu", _relu_forward, _relu_backward)
ActivationRegistry.register("leaky_relu", _leaky_relu_forward, _leaky_relu_backward)
ActivationRegistry.register("sigmoid", _sigmoid_forward, _sigmoid_backward)
ActivationRegistry.register("tanh", _tanh_forward, _tanh_backward)
ActivationRegistry.register("softmax", _softmax_forward, _softmax_backward)
ActivationRegistry.register("elu", _elu_forward, _elu_backward)
ActivationRegistry.register("selu", _selu_forward, _selu_backward)
ActivationRegistry.register("swish", _swish_forward, _swish_backward)
ActivationRegistry.register("silu", _swish_forward, _swish_backward)
ActivationRegistry.register("gelu", _gelu_forward, _gelu_backward)
ActivationRegistry.register("softplus", _softplus_forward, _softplus_backward)
ActivationRegistry.register("softsign", _softsign_forward, _softsign_backward)
ActivationRegistry.register("linear", _linear_forward, _linear_backward)
ActivationRegistry.register("identity", _linear_forward, _linear_backward)
ActivationRegistry.register("mish", _mish_forward, _mish_backward)


# =============================================================================
# Regularizer Classes
# =============================================================================


class Regularizer:
    """Base class for regularizers."""

    def __init__(self, l1: float = 0.0, l2: float = 0.0):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, x: np.ndarray) -> float:
        """Compute regularization loss."""
        loss = 0.0
        if self.l1 > 0:
            loss += self.l1 * np.sum(np.abs(x))
        if self.l2 > 0:
            loss += self.l2 * np.sum(x**2)
        return loss

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute regularization gradient."""
        grad = np.zeros_like(x)
        if self.l1 > 0:
            grad += self.l1 * np.sign(x)
        if self.l2 > 0:
            grad += self.l2 * 2.0 * x
        return grad


class L1Regularizer(Regularizer):
    """L1 (Lasso) regularizer."""

    def __init__(self, l1: float = 0.01):
        super().__init__(l1=l1, l2=0.0)


class L2Regularizer(Regularizer):
    """L2 (Ridge) regularizer."""

    def __init__(self, l2: float = 0.01):
        super().__init__(l1=0.0, l2=l2)


class L1L2Regularizer(Regularizer):
    """Combined L1 and L2 (Elastic Net) regularizer."""

    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        super().__init__(l1=l1, l2=l2)


# =============================================================================
# Constraint Classes
# =============================================================================


class Constraint:
    """Base class for weight constraints."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply constraint to weights."""
        return x


class MaxNorm(Constraint):
    """Max norm constraint."""

    def __init__(self, max_value: float = 2.0, axis: int = 0):
        self.max_value = max_value
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        norms = np.sqrt(np.sum(x**2, axis=self.axis, keepdims=True))
        desired = np.clip(norms, 0, self.max_value)
        return x * (desired / (norms + 1e-7))


class NonNeg(Constraint):
    """Non-negative constraint."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)


class UnitNorm(Constraint):
    """Unit norm constraint."""

    def __init__(self, axis: int = 0):
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x / (np.sqrt(np.sum(x**2, axis=self.axis, keepdims=True)) + 1e-7)


class MinMaxNorm(Constraint):
    """Min-max norm constraint."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0, axis: int = 0):
        self.min_value = min_value
        self.max_value = max_value
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        norms = np.sqrt(np.sum(x**2, axis=self.axis, keepdims=True))
        desired = np.clip(norms, self.min_value, self.max_value)
        return x * (desired / (norms + 1e-7))


# =============================================================================
# Dense Layer
# =============================================================================


class Dense(BaseLayer):
    """
    Universal Dense (Fully Connected) Layer.

    A highly configurable dense layer supporting arbitrary dtypes,
    custom initializers, activations, regularizers, and constraints.

    Parameters
    ----------
    units : int
        Number of neurons (output dimension). Must be positive.
    activation : str or callable, optional
        Activation function. Can be:
        - String: 'relu', 'sigmoid', 'tanh', 'softmax', 'elu', 'selu',
                  'swish', 'silu', 'gelu', 'softplus', 'softsign', 'mish', 'linear'
        - Callable: Custom activation function with forward(z) and backward(grad, z, output)
        - None: No activation (linear)
    use_bias : bool, default=True
        Whether to include a bias term.
    kernel_initializer : str or Initializer, default='he_normal'
        Initializer for weight matrix. Options:
        - String: 'he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform',
                  'lecun_normal', 'lecun_uniform', 'zeros', 'ones', 'random_normal'
        - Initializer: Custom initializer instance
    bias_initializer : str or Initializer, default='zeros'
        Initializer for bias vector.
    kernel_regularizer : Regularizer or dict, optional
        Weight regularizer. Can be:
        - Regularizer instance
        - Dict: {'l1': 0.01, 'l2': 0.01}
        - String: 'l1', 'l2', 'l1_l2'
    bias_regularizer : Regularizer or dict, optional
        Bias regularizer.
    activity_regularizer : Regularizer or dict, optional
        Activity (output) regularizer.
    kernel_constraint : Constraint, optional
        Constraint applied to weights after update.
    bias_constraint : Constraint, optional
        Constraint applied to bias after update.
    dtype : str or np.dtype, default='float32'
        Data type for computations. Options:
        - 'float16', 'half': 16-bit floating point
        - 'float32', 'single': 32-bit floating point (default)
        - 'float64', 'double': 64-bit floating point
        - 'bfloat16', 'bf16': Brain float 16 (requires ml_dtypes)
    compute_dtype : str or np.dtype, optional
        Data type for internal computations. If None, uses dtype.
        Useful for mixed precision training (e.g., dtype='float16', compute_dtype='float32').
    input_shape : tuple, optional
        Input shape for the layer. Used for building layer without data.
    name : str, optional
        Layer name.

    Attributes
    ----------
    weights : ndarray
        Weight matrix of shape (input_dim, units).
    bias : ndarray
        Bias vector of shape (units,) if use_bias=True.
    grad_weights : ndarray
        Gradient of loss w.r.t. weights.
    grad_bias : ndarray
        Gradient of loss w.r.t. bias.

    Examples
    --------
    >>> # Basic usage
    >>> layer = Dense(128, activation='relu')

    >>> # With all customizations
    >>> layer = Dense(
    ...     units=256,
    ...     activation='gelu',
    ...     use_bias=True,
    ...     kernel_initializer='xavier_uniform',
    ...     bias_initializer='zeros',
    ...     kernel_regularizer={'l2': 0.01},
    ...     kernel_constraint=MaxNorm(3.0),
    ...     dtype='float32'
    ... )

    >>> # Mixed precision training
    >>> layer = Dense(128, dtype='float16', compute_dtype='float32')

    >>> # Custom activation
    >>> def my_activation(z):
    ...     return z * (z > 0)  # Custom ReLU variant
    >>> layer = Dense(64, activation=my_activation)

    Notes
    -----
    For best CPU performance, use float32 (default). Float64 provides
    higher precision but is ~2x slower due to memory bandwidth. Float16
    and BF16 may not provide speedup on CPU without hardware acceleration.

    The layer automatically handles dtype conversions and ensures
    all operations use the specified dtype for consistency.
    """

    # Use __slots__ for memory efficiency
    __slots__ = (
        "units",
        "activation",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "_dtype",
        "_compute_dtype",
        "_input_shape_arg",
        "weights",
        "bias",
        "grad_weights",
        "grad_bias",
        "_input_cache",
        "_z_cache",
        "_output_cache",
        "_activation_forward",
        "_activation_backward",
        "_activation_name",
        "_activation_params",
        "_inv_batch_size",
    )

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, BaseInitializer] = "he_normal",
        bias_initializer: Union[str, BaseInitializer] = "zeros",
        kernel_regularizer: Optional[Union[Regularizer, Dict, str]] = None,
        bias_regularizer: Optional[Union[Regularizer, Dict, str]] = None,
        activity_regularizer: Optional[Union[Regularizer, Dict, str]] = None,
        kernel_constraint: Optional[Constraint] = None,
        bias_constraint: Optional[Constraint] = None,
        dtype: Union[str, np.dtype] = "float32",
        compute_dtype: Optional[Union[str, np.dtype]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        # Validate units
        if not isinstance(units, int) or units <= 0:
            raise ValueError(f"units must be a positive integer, got {units}")

        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self._input_shape_arg = input_shape

        # Set up dtypes
        self._dtype = self._parse_dtype(dtype)
        self._compute_dtype = (
            self._parse_dtype(compute_dtype) if compute_dtype else self._dtype
        )

        # Set up activation
        self._activation_name = None
        self._activation_params = {}
        self._activation_forward = None
        self._activation_backward = None
        self._setup_activation(activation)
        self.activation = activation

        # Set up regularizers
        self.kernel_regularizer = self._parse_regularizer(kernel_regularizer)
        self.bias_regularizer = self._parse_regularizer(bias_regularizer)
        self.activity_regularizer = self._parse_regularizer(activity_regularizer)

        # Parameters (initialized in build)
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        # Gradients
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_bias: Optional[np.ndarray] = None

        # Caches for backward pass
        self._input_cache: Optional[np.ndarray] = None
        self._z_cache: Optional[np.ndarray] = None
        self._output_cache: Optional[np.ndarray] = None
        self._inv_batch_size: float = 0.0

    @staticmethod
    def _parse_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
        """Parse dtype string or object into numpy dtype."""
        if isinstance(dtype, np.dtype):
            return dtype

        dtype_str = str(dtype).lower()

        if dtype_str in DTYPE_MAP:
            return DTYPE_MAP[dtype_str]

        # Try to get from numpy directly
        try:
            return np.dtype(dtype)
        except TypeError:
            raise ValueError(
                f"Unknown dtype: '{dtype}'. Supported: {list(DTYPE_MAP.keys())}"
            )

    def _parse_regularizer(
        self, reg: Optional[Union[Regularizer, Dict, str]]
    ) -> Optional[Regularizer]:
        """Parse regularizer specification."""
        if reg is None:
            return None
        if isinstance(reg, Regularizer):
            return reg
        if isinstance(reg, dict):
            return Regularizer(l1=reg.get("l1", 0.0), l2=reg.get("l2", 0.0))
        if isinstance(reg, str):
            reg_lower = reg.lower()
            if reg_lower == "l1":
                return L1Regularizer()
            elif reg_lower == "l2":
                return L2Regularizer()
            elif reg_lower in ("l1_l2", "elastic_net"):
                return L1L2Regularizer()
            else:
                raise ValueError(f"Unknown regularizer: '{reg}'")
        raise TypeError(f"Invalid regularizer type: {type(reg)}")

    def _setup_activation(self, activation: Optional[Union[str, Callable]]) -> None:
        """Set up activation function."""
        if activation is None or activation == "linear":
            self._activation_forward = _linear_forward
            self._activation_backward = _linear_backward
            self._activation_name = "linear"
            return

        if isinstance(activation, str):
            # Parse activation string (may include parameters)
            act_lower = activation.lower()

            # Handle leaky_relu with alpha parameter
            if act_lower.startswith("leaky_relu"):
                self._activation_name = "leaky_relu"
                # Parse alpha if provided: "leaky_relu_0.2" or "leaky_relu(alpha=0.2)"
                alpha = 0.01  # default
                if "_" in act_lower:
                    try:
                        alpha = float(act_lower.split("_")[-1])
                    except ValueError:
                        pass
                self._activation_params = {"alpha": alpha}
                self._activation_forward = lambda z, a=alpha: _leaky_relu_forward(z, a)
                self._activation_backward = lambda g, z, o, a=alpha: (
                    _leaky_relu_backward(g, z, o, a)
                )
                return

            # Handle ELU with alpha parameter
            if (
                act_lower.startswith("elu")
                and act_lower != "elu"
                and act_lower != "selu"
            ):
                self._activation_name = "elu"
                alpha = 1.0
                if "_" in act_lower:
                    try:
                        alpha = float(act_lower.split("_")[-1])
                    except ValueError:
                        pass
                self._activation_params = {"alpha": alpha}
                self._activation_forward = lambda z, a=alpha: _elu_forward(z, a)
                self._activation_backward = lambda g, z, o, a=alpha: _elu_backward(
                    g, z, o, a
                )
                return

            # Look up in registry
            registered = ActivationRegistry.get(act_lower)
            if registered:
                self._activation_forward = registered["forward"]
                self._activation_backward = registered["backward"]
                self._activation_name = act_lower
                return

            raise ValueError(
                f"Unknown activation: '{activation}'. "
                f"Available: {ActivationRegistry.list_available()}"
            )

        if callable(activation):
            # Custom callable activation
            self._activation_forward = activation
            # Try to get backward method
            if hasattr(activation, "backward"):
                self._activation_backward = activation.backward
            elif hasattr(activation, "gradient"):
                self._activation_backward = lambda g, z, o: activation.gradient(g, z)
            else:
                # No backward provided, assume no gradient modification
                warnings.warn(
                    f"Activation {activation} has no backward/gradient method. "
                    "Using identity gradient."
                )
                self._activation_backward = _linear_backward
            self._activation_name = "custom"
            return

        raise TypeError(f"activation must be str or callable, got {type(activation)}")

    @property
    def dtype(self) -> np.dtype:
        """Get the layer's data type."""
        return self._dtype

    @property
    def compute_dtype(self) -> np.dtype:
        """Get the layer's compute data type."""
        return self._compute_dtype

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """
        Build the layer by initializing parameters.

        Parameters
        ----------
        input_shape : tuple
            Shape of input (batch_size, input_dim) or (input_dim,)
        """
        if self.built:
            return

        # Handle input shape
        if len(input_shape) < 1:
            raise ValueError(
                f"input_shape must have at least 1 dimension, got {input_shape}"
            )

        input_dim = input_shape[-1]
        self._input_shape = input_shape
        self._output_shape = (
            (input_shape[0], self.units) if len(input_shape) > 1 else (self.units,)
        )

        # Initialize weights
        initializer = get_initializer(self.kernel_initializer)
        self.weights = initializer.initialize((input_dim, self.units)).astype(
            self._dtype
        )

        # Initialize bias
        if self.use_bias:
            bias_init = get_initializer(self.bias_initializer)
            self.bias = bias_init.initialize((self.units,)).astype(self._dtype)
        else:
            self.bias = None

        # Initialize gradient buffers
        self.grad_weights = np.zeros_like(self.weights)
        if self.use_bias:
            self.grad_bias = np.zeros_like(self.bias)

        self.built = True

    def _ensure_dtype(self, x: np.ndarray) -> np.ndarray:
        """Ensure input is correct dtype and contiguous."""
        if x.dtype != self._dtype:
            x = x.astype(self._dtype)
        if not x.flags["C_CONTIGUOUS"]:
            x = np.ascontiguousarray(x)
        return x

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the layer.

        Parameters
        ----------
        x : ndarray
            Input of shape (batch_size, input_dim)
        training : bool
            Whether in training mode (affects dropout, batchnorm, etc.)

        Returns
        -------
        output : ndarray
            Output of shape (batch_size, units)
        """
        # Build if needed
        if not self.built:
            self.build((None,) + x.shape[1:] if x.ndim > 1 else (x.shape[-1],))

        # Ensure correct dtype
        x = self._ensure_dtype(x)

        # Cache for backward pass
        self._input_cache = x

        # Linear transformation: z = x @ W + b
        z = x @ self.weights
        if self.use_bias:
            z = z + self.bias

        # Cache pre-activation
        self._z_cache = z

        # Apply activation
        output = self._activation_forward(z)
        self._output_cache = output

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass to compute gradients.

        Parameters
        ----------
        grad_output : ndarray
            Gradient from next layer, shape (batch_size, units)

        Returns
        -------
        grad_input : ndarray
            Gradient for previous layer, shape (batch_size, input_dim)
        """
        if self._input_cache is None:
            raise RuntimeError("Forward pass must be called before backward pass")

        # Ensure correct dtype
        grad_output = self._ensure_dtype(grad_output)

        batch_size = grad_output.shape[0]
        self._inv_batch_size = 1.0 / batch_size

        # Gradient through activation
        grad_z = self._activation_backward(
            grad_output, self._z_cache, self._output_cache
        )

        # Compute weight gradients: dW = X.T @ grad_z / batch_size
        np.dot(self._input_cache.T, grad_z, out=self.grad_weights)
        self.grad_weights *= self._inv_batch_size

        # Add regularization gradient
        if self.kernel_regularizer is not None:
            self.grad_weights += self.kernel_regularizer.gradient(self.weights)

        # Compute bias gradients: db = mean(grad_z, axis=0)
        if self.use_bias:
            np.mean(grad_z, axis=0, out=self.grad_bias)
            if self.bias_regularizer is not None:
                self.grad_bias += self.bias_regularizer.gradient(self.bias)

        # Compute input gradients: dX = grad_z @ W.T
        grad_input = grad_z @ self.weights.T

        return grad_input

    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        params = {"weights": self.weights}
        if self.use_bias:
            params["bias"] = self.bias
        return params

    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set layer parameters."""
        if "weights" in params:
            self.weights = self._ensure_dtype(np.asarray(params["weights"]))
        if self.use_bias and "bias" in params:
            self.bias = self._ensure_dtype(np.asarray(params["bias"]))

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get parameter gradients."""
        grads = {"weights": self.grad_weights}
        if self.use_bias:
            grads["bias"] = self.grad_bias
        return grads

    def apply_constraints(self) -> None:
        """Apply constraints to weights and bias."""
        if self.kernel_constraint is not None and self.weights is not None:
            self.weights = self.kernel_constraint(self.weights)
        if self.bias_constraint is not None and self.bias is not None:
            self.bias = self.bias_constraint(self.bias)

    def get_regularization_loss(self) -> float:
        """Get total regularization loss."""
        loss = 0.0
        if self.kernel_regularizer is not None and self.weights is not None:
            loss += self.kernel_regularizer(self.weights)
        if self.bias_regularizer is not None and self.bias is not None:
            loss += self.bias_regularizer(self.bias)
        return loss

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = {
            "units": self.units,
            "activation": self._activation_name,
            "activation_params": self._activation_params,
            "use_bias": self.use_bias,
            "dtype": str(self._dtype),
            "compute_dtype": str(self._compute_dtype),
        }

        # Add initializer info
        if isinstance(self.kernel_initializer, str):
            config["kernel_initializer"] = self.kernel_initializer
        if isinstance(self.bias_initializer, str):
            config["bias_initializer"] = self.bias_initializer

        # Add regularizer info
        if self.kernel_regularizer is not None:
            config["kernel_regularizer"] = {
                "l1": self.kernel_regularizer.l1,
                "l2": self.kernel_regularizer.l2,
            }

        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Dense":
        """Create layer from configuration."""
        # Reconstruct activation
        activation = config.get("activation", "linear")
        if activation == "leaky_relu" and "activation_params" in config:
            alpha = config["activation_params"].get("alpha", 0.01)
            activation = f"leaky_relu_{alpha}"

        # Reconstruct regularizer
        kernel_regularizer = None
        if "kernel_regularizer" in config:
            kernel_regularizer = Regularizer(**config["kernel_regularizer"])

        return cls(
            units=config["units"],
            activation=activation,
            use_bias=config.get("use_bias", True),
            kernel_initializer=config.get("kernel_initializer", "he_normal"),
            bias_initializer=config.get("bias_initializer", "zeros"),
            kernel_regularizer=kernel_regularizer,
            dtype=config.get("dtype", "float32"),
            compute_dtype=config.get("compute_dtype"),
            name=config.get("name"),
        )

    def __repr__(self) -> str:
        parts = [f"units={self.units}"]
        if self._activation_name and self._activation_name != "linear":
            parts.append(f"activation='{self._activation_name}'")
        parts.append(f"dtype='{self._dtype}'")
        return f"Dense({', '.join(parts)})"


# =============================================================================
# Convenience Aliases
# =============================================================================


class DenseFloat16(Dense):
    """Dense layer with float16 precision."""

    def __init__(self, units: int, **kwargs):
        kwargs["dtype"] = "float16"
        super().__init__(units, **kwargs)


class DenseFloat32(Dense):
    """Dense layer with float32 precision (default)."""

    def __init__(self, units: int, **kwargs):
        kwargs["dtype"] = "float32"
        super().__init__(units, **kwargs)


class DenseFloat64(Dense):
    """Dense layer with float64 precision."""

    def __init__(self, units: int, **kwargs):
        kwargs["dtype"] = "float64"
        super().__init__(units, **kwargs)


class DenseBF16(Dense):
    """Dense layer with bfloat16 precision."""

    def __init__(self, units: int, **kwargs):
        if not BF16_AVAILABLE:
            raise ImportError(
                "bfloat16 requires ml_dtypes. Install with: pip install ml_dtypes"
            )
        kwargs["dtype"] = "bfloat16"
        super().__init__(units, **kwargs)


# Mixed precision alias
class DenseMixedPrecision(Dense):
    """
    Dense layer with mixed precision training.

    Uses lower precision for storage and higher precision for computation.
    """

    def __init__(
        self,
        units: int,
        storage_dtype: str = "float16",
        compute_dtype: str = "float32",
        **kwargs,
    ):
        kwargs["dtype"] = storage_dtype
        kwargs["compute_dtype"] = compute_dtype
        super().__init__(units, **kwargs)
