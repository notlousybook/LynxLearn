"""
Weight initializers for neural network layers.

This module provides various weight initialization strategies
to help with training stability and convergence.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class BaseInitializer(ABC):
    """
    Base class for all weight initializers.

    All initializer implementations must inherit from this class
    and implement the initialize method.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> initializer = HeInitializer(seed=42)
    >>> weights = initializer.initialize((784, 128))
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng = np.random.RandomState(seed)

    @abstractmethod
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Initialize weights with the given shape.

        Parameters
        ----------
        shape : tuple
            Shape of the weight array to initialize

        Returns
        -------
        weights : ndarray
            Initialized weight array
        """
        pass

    def reset(self) -> None:
        """Reset the random state."""
        self._rng = np.random.RandomState(self.seed)


class RandomNormal(BaseInitializer):
    """
    Initialize weights from a normal (Gaussian) distribution.

    Parameters
    ----------
    mean : float, default=0.0
        Mean of the distribution
    stddev : float, default=0.05
        Standard deviation of the distribution
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> initializer = RandomNormal(mean=0.0, stddev=0.05)
    >>> weights = initializer.initialize((100, 50))
    """

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights from normal distribution."""
        return self._rng.normal(self.mean, self.stddev, shape)


class RandomUniform(BaseInitializer):
    """
    Initialize weights from a uniform distribution.

    Parameters
    ----------
    minval : float, default=-0.05
        Lower bound of the distribution
    maxval : float, default=0.05
        Upper bound of the distribution
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> initializer = RandomUniform(minval=-0.05, maxval=0.05)
    >>> weights = initializer.initialize((100, 50))
    """

    def __init__(
        self,
        minval: float = -0.05,
        maxval: float = 0.05,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)
        self.minval = minval
        self.maxval = maxval

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights from uniform distribution."""
        return self._rng.uniform(self.minval, self.maxval, shape)


class Zeros(BaseInitializer):
    """
    Initialize weights with zeros.

    Typically used for bias initialization.

    Examples
    --------
    >>> initializer = Zeros()
    >>> bias = initializer.initialize((128,))
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights with zeros."""
        return np.zeros(shape)


class Ones(BaseInitializer):
    """
    Initialize weights with ones.

    Examples
    --------
    >>> initializer = Ones()
    >>> weights = initializer.initialize((128,))
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights with ones."""
        return np.ones(shape)


class Constant(BaseInitializer):
    """
    Initialize weights with a constant value.

    Parameters
    ----------
    value : float, default=0.0
        The constant value to use

    Examples
    --------
    >>> initializer = Constant(value=0.5)
    >>> weights = initializer.initialize((100, 50))
    """

    def __init__(self, value: float = 0.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.value = value

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights with constant value."""
        return np.full(shape, self.value)


class XavierNormal(BaseInitializer):
    """
    Xavier/Glorot normal initialization.

    Designed for use with tanh and sigmoid activations.
    Draws samples from a normal distribution with:
        stddev = sqrt(2 / (fan_in + fan_out))

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] Glorot & Bengio, "Understanding the difficulty of training
           deep feedforward neural networks", AISTATS 2010

    Examples
    --------
    >>> initializer = XavierNormal(seed=42)
    >>> weights = initializer.initialize((784, 128))  # For tanh/sigmoid
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using Xavier normal distribution."""
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return self._rng.normal(0.0, stddev, shape)


class XavierUniform(BaseInitializer):
    """
    Xavier/Glorot uniform initialization.

    Designed for use with tanh and sigmoid activations.
    Draws samples from a uniform distribution with:
        limit = sqrt(6 / (fan_in + fan_out))

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] Glorot & Bengio, "Understanding the difficulty of training
           deep feedforward neural networks", AISTATS 2010

    Examples
    --------
    >>> initializer = XavierUniform(seed=42)
    >>> weights = initializer.initialize((784, 128))  # For tanh/sigmoid
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using Xavier uniform distribution."""
        fan_in = shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self._rng.uniform(-limit, limit, shape)


class HeNormal(BaseInitializer):
    """
    He/Kaiming normal initialization.

    Designed for use with ReLU and its variants (LeakyReLU, PReLU, etc.).
    Draws samples from a normal distribution with:
        stddev = sqrt(2 / fan_in)

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] He et al., "Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification", ICCV 2015

    Examples
    --------
    >>> initializer = HeNormal(seed=42)
    >>> weights = initializer.initialize((784, 128))  # For ReLU
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using He normal distribution."""
        fan_in = shape[0]
        stddev = np.sqrt(2.0 / fan_in)
        return self._rng.normal(0.0, stddev, shape)


class HeUniform(BaseInitializer):
    """
    He/Kaiming uniform initialization.

    Designed for use with ReLU and its variants (LeakyReLU, PReLU, etc.).
    Draws samples from a uniform distribution with:
        limit = sqrt(6 / fan_in)

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] He et al., "Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification", ICCV 2015

    Examples
    --------
    >>> initializer = HeUniform(seed=42)
    >>> weights = initializer.initialize((784, 128))  # For ReLU
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using He uniform distribution."""
        fan_in = shape[0]
        limit = np.sqrt(6.0 / fan_in)
        return self._rng.uniform(-limit, limit, shape)


class LeCunNormal(BaseInitializer):
    """
    LeCun normal initialization.

    Designed for use with SELU activation in self-normalizing networks.
    Draws samples from a normal distribution with:
        stddev = sqrt(1 / fan_in)

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] LeCun et al., "Efficient BackProp", Neural Networks:
           Tricks of the Trade, 1998

    Examples
    --------
    >>> initializer = LeCunNormal(seed=42)
    >>> weights = initializer.initialize((784, 128))  # For SELU
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using LeCun normal distribution."""
        fan_in = shape[0]
        stddev = np.sqrt(1.0 / fan_in)
        return self._rng.normal(0.0, stddev, shape)


class LeCunUniform(BaseInitializer):
    """
    LeCun uniform initialization.

    Designed for use with SELU activation.
    Draws samples from a uniform distribution with:
        limit = sqrt(3 / fan_in)

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] LeCun et al., "Efficient BackProp", Neural Networks:
           Tricks of the Trade, 1998

    Examples
    --------
    >>> initializer = LeCunUniform(seed=42)
    >>> weights = initializer.initialize((784, 128))
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights using LeCun uniform distribution."""
        fan_in = shape[0]
        limit = np.sqrt(3.0 / fan_in)
        return self._rng.uniform(-limit, limit, shape)


class Orthogonal(BaseInitializer):
    """
    Orthogonal initialization.

    Generates a random orthogonal matrix. Useful for deep networks
    to prevent vanishing/exploding gradients.

    Parameters
    ----------
    gain : float, default=1.0
        Multiplicative factor to apply to the orthogonal matrix
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    .. [1] Saxe et al., "Exact solutions to the nonlinear dynamics of
           learning in deep linear neural networks", ICLR 2014

    Examples
    --------
    >>> initializer = Orthogonal(gain=1.0)
    >>> weights = initializer.initialize((784, 128))
    """

    def __init__(self, gain: float = 1.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.gain = gain

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights with orthogonal matrix."""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = self._rng.normal(0.0, 1.0, flat_shape)

        # QR decomposition to get orthogonal matrix
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v

        # Reshape and apply gain
        return self.gain * q.reshape(shape)


class TruncatedNormal(BaseInitializer):
    """
    Initialize weights from a truncated normal distribution.

    Values are drawn from a normal distribution but values more than
    two standard deviations from the mean are discarded and re-drawn.

    Parameters
    ----------
    mean : float, default=0.0
        Mean of the distribution
    stddev : float, default=0.05
        Standard deviation of the distribution
    seed : int, optional
        Random seed for reproducibility

    Examples
    --------
    >>> initializer = TruncatedNormal(mean=0.0, stddev=0.05)
    >>> weights = initializer.initialize((100, 50))
    """

    def __init__(
        self, mean: float = 0.0, stddev: float = 0.05, seed: Optional[int] = None
    ):
        super().__init__(seed)
        self.mean = mean
        self.stddev = stddev

    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Initialize weights from truncated normal distribution."""
        # Use scipy-like truncated normal
        a = -2.0  # Lower bound in standard units
        b = 2.0  # Upper bound in standard units

        # Generate samples until we have enough
        result = []
        total_size = np.prod(shape)

        while len(result) < total_size:
            samples = self._rng.normal(self.mean, self.stddev, size=(total_size * 2,))
            # Filter samples within bounds
            lower = self.mean + a * self.stddev
            upper = self.mean + b * self.stddev
            valid = samples[(samples >= lower) & (samples <= upper)]
            result.extend(valid[: total_size - len(result)])

        return np.array(result[:total_size]).reshape(shape)


# Aliases for common initializers
GlorotNormal = XavierNormal
GlorotUniform = XavierUniform
KaimingNormal = HeNormal
KaimingUniform = HeUniform


# Registry for string-based initialization
INITIALIZERS = {
    "zeros": Zeros,
    "ones": Ones,
    "constant": Constant,
    "random_normal": RandomNormal,
    "random_uniform": RandomUniform,
    "truncated_normal": TruncatedNormal,
    "xavier": XavierNormal,
    "xavier_normal": XavierNormal,
    "xavier_uniform": XavierUniform,
    "glorot": XavierNormal,
    "glorot_normal": XavierNormal,
    "glorot_uniform": XavierUniform,
    "he": HeNormal,
    "he_normal": HeNormal,
    "he_uniform": HeUniform,
    "kaiming": HeNormal,
    "kaiming_normal": HeNormal,
    "kaiming_uniform": HeUniform,
    "lecun": LeCunNormal,
    "lecun_normal": LeCunNormal,
    "lecun_uniform": LeCunUniform,
    "orthogonal": Orthogonal,
}


def get_initializer(
    initializer: Union[str, BaseInitializer, None], seed: Optional[int] = None
) -> BaseInitializer:
    """
    Get an initializer instance from string or return the provided initializer.

    Parameters
    ----------
    initializer : str, BaseInitializer, or None
        Initializer name, instance, or None (defaults to HeNormal)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    initializer : BaseInitializer
        Initializer instance

    Examples
    --------
    >>> init = get_initializer('he_normal', seed=42)
    >>> init = get_initializer('xavier')
    >>> init = get_initializer(None)  # Returns HeNormal
    """
    if initializer is None:
        return HeNormal(seed=seed)

    if isinstance(initializer, BaseInitializer):
        return initializer

    if isinstance(initializer, str):
        name = initializer.lower().replace("-", "_")
        if name not in INITIALIZERS:
            raise ValueError(
                f"Unknown initializer: '{initializer}'. "
                f"Available: {list(INITIALIZERS.keys())}"
            )
        return INITIALIZERS[name](seed=seed)

    raise TypeError(
        f"initializer must be str, BaseInitializer, or None, got {type(initializer)}"
    )
