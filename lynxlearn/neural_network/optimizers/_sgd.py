"""
Stochastic Gradient Descent (SGD) optimizer implementation.
"""

from typing import Any, Dict, Optional

import numpy as np

from ._base import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer with momentum support.

    Implements vanilla SGD, momentum, and Nesterov accelerated gradient.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate for parameter updates
    momentum : float, default=0.0
        Momentum coefficient. Use 0.0 for vanilla SGD.
        Typical values: 0.9, 0.99
    nesterov : bool, default=False
        Whether to use Nesterov momentum (NAG).
        Requires momentum > 0 to have effect.
    clipnorm : float, optional
        Gradient clipping by global norm. If set, gradients
        are scaled to have max norm equal to this value.
    clipvalue : float, optional
        Gradient clipping by value. Gradients are clipped
        to [-clipvalue, clipvalue].

    Attributes
    ----------
    velocities : dict
        Velocity arrays for each layer (used with momentum)

    Examples
    --------
    >>> # Vanilla SGD
    >>> optimizer = SGD(learning_rate=0.01)

    >>> # SGD with momentum
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9)

    >>> # SGD with Nesterov momentum
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

    >>> # With gradient clipping
    >>> optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)

    References
    ----------
    .. [1] Robbins & Monro, "A Stochastic Approximation Method", 1951
    .. [2] Polyak, "Some methods of speeding up the convergence of
           iteration methods", 1964 (momentum)
    .. [3] Nesterov, "A method for unconstrained convex minimization
           problem with the rate of convergence O(1/k^2)", 1983
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        clipnorm: Optional[float] = None,
        clipvalue: Optional[float] = None,
    ):
        super().__init__(learning_rate)

        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {momentum}")

        self.momentum = momentum
        self.nesterov = nesterov
        self.clipnorm = clipnorm
        self.clipvalue = clipvalue

        # Velocity storage for momentum
        self._velocities: Dict[int, Dict[str, np.ndarray]] = {}

    def update(self, layer: Any) -> None:
        """
        Update layer parameters using SGD with optional momentum.

        Parameters
        ----------
        layer : BaseLayer
            Layer with parameters to update. Must implement
            get_params(), set_params(), and get_gradients() methods.
        """
        # Get layer parameters and gradients
        params = layer.get_params()
        grads = layer.get_gradients()

        if not params or not grads:
            return

        # Apply gradient clipping
        grads = self._clip_gradients(grads)

        # Get layer ID for velocity storage
        layer_id = id(layer)

        # Initialize velocities if needed
        if layer_id not in self._velocities:
            self._velocities[layer_id] = {}
            for key in grads:
                self._velocities[layer_id][key] = np.zeros_like(grads[key])

        # Update each parameter
        updated_params = {}
        for key in params:
            param = params[key]
            grad = grads[key]

            if self.momentum > 0.0:
                # Get velocity for this parameter
                velocity = self._velocities[layer_id][key]

                # Update velocity: v = momentum * v - lr * grad
                velocity = self.momentum * velocity - self.learning_rate * grad

                # Store updated velocity
                self._velocities[layer_id][key] = velocity

                if self.nesterov:
                    # Nesterov: param += momentum * velocity - lr * grad
                    updated_params[key] = (
                        param + self.momentum * velocity - self.learning_rate * grad
                    )
                else:
                    # Standard momentum: param += velocity
                    updated_params[key] = param + velocity
            else:
                # Vanilla SGD: param -= lr * grad
                updated_params[key] = param - self.learning_rate * grad

        # Set updated parameters
        layer.set_params(updated_params)

        # Increment iteration counter
        self.iterations += 1

    def _clip_gradients(self, grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply gradient clipping.

        Parameters
        ----------
        grads : dict
            Dictionary of gradient arrays

        Returns
        -------
        clipped_grads : dict
            Dictionary of clipped gradient arrays
        """
        if not grads:
            return grads

        # Clip by global norm
        if self.clipnorm is not None and self.clipnorm > 0:
            # Compute global norm
            global_norm = np.sqrt(sum(np.sum(np.square(g)) for g in grads.values()))

            if global_norm > self.clipnorm:
                scale = self.clipnorm / (global_norm + 1e-7)
                grads = {k: v * scale for k, v in grads.items()}

        # Clip by value
        if self.clipvalue is not None and self.clipvalue > 0:
            grads = {
                k: np.clip(v, -self.clipvalue, self.clipvalue) for k, v in grads.items()
            }

        return grads

    def get_state(self) -> Dict[str, Any]:
        """
        Get optimizer state for serialization.

        Returns
        -------
        state : dict
            Dictionary containing optimizer state variables
        """
        state = super().get_state()
        state.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                "clipnorm": self.clipnorm,
                "clipvalue": self.clipvalue,
                "velocities": self._velocities.copy(),
            }
        )
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore optimizer state from dictionary.

        Parameters
        ----------
        state : dict
            Dictionary containing optimizer state variables
        """
        super().set_state(state)
        self.momentum = state.get("momentum", self.momentum)
        self.nesterov = state.get("nesterov", self.nesterov)
        self.clipnorm = state.get("clipnorm", self.clipnorm)
        self.clipvalue = state.get("clipvalue", self.clipvalue)
        self._velocities = state.get("velocities", {})

    def reset(self) -> None:
        """
        Reset optimizer state.

        Clears velocity history and iteration counter.
        """
        super().reset()
        self._velocities.clear()

    def get_config(self) -> Dict[str, Any]:
        """
        Get optimizer configuration for serialization.

        Returns
        -------
        config : dict
            Dictionary of optimizer configuration
        """
        config = super().get_config()
        config.update(
            {
                "momentum": self.momentum,
                "nesterov": self.nesterov,
                "clipnorm": self.clipnorm,
                "clipvalue": self.clipvalue,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SGD":
        """
        Create SGD optimizer from configuration dictionary.

        Parameters
        ----------
        config : dict
            Dictionary of optimizer configuration

        Returns
        -------
        optimizer : SGD
            New SGD optimizer instance
        """
        return cls(**config)

    def __repr__(self) -> str:
        parts = [f"learning_rate={self.learning_rate}"]
        if self.momentum > 0:
            parts.append(f"momentum={self.momentum}")
            if self.nesterov:
                parts.append("nesterov=True")
        if self.clipnorm is not None:
            parts.append(f"clipnorm={self.clipnorm}")
        if self.clipvalue is not None:
            parts.append(f"clipvalue={self.clipvalue}")
        return f"SGD({', '.join(parts)})"
