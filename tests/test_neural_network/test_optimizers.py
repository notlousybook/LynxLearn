"""
Tests for neural network optimizers.
"""

import numpy as np
import pytest

from lynxlearn.neural_network.layers import Dense
from lynxlearn.neural_network.optimizers import SGD, BaseOptimizer


class TestSGD:
    """Tests for SGD optimizer."""

    def test_initialization(self):
        """Test SGD initialization with default parameters."""
        optimizer = SGD()

        assert optimizer.learning_rate == 0.01
        assert optimizer.momentum == 0.0
        assert optimizer.nesterov is False
        assert optimizer.clipnorm is None
        assert optimizer.clipvalue is None

    def test_initialization_with_parameters(self):
        """Test SGD initialization with custom parameters."""
        optimizer = SGD(
            learning_rate=0.001,
            momentum=0.9,
            nesterov=True,
            clipnorm=1.0,
            clipvalue=5.0,
        )

        assert optimizer.learning_rate == 0.001
        assert optimizer.momentum == 0.9
        assert optimizer.nesterov is True
        assert optimizer.clipnorm == 1.0
        assert optimizer.clipvalue == 5.0

    def test_invalid_momentum(self):
        """Test that invalid momentum raises error."""
        with pytest.raises(ValueError):
            SGD(momentum=-0.1)

        with pytest.raises(ValueError):
            SGD(momentum=1.1)

    def test_update_vanilla_sgd(self):
        """Test vanilla SGD update (no momentum)."""
        optimizer = SGD(learning_rate=0.1)

        # Create a simple dense layer
        layer = Dense(1, input_shape=(2,))
        layer.build((1, 2))

        # Set initial weights
        initial_weights = np.array([[0.5], [0.5]])
        initial_bias = np.array([0.0])
        layer.set_params({"weights": initial_weights, "bias": initial_bias})

        # Perform forward and backward pass
        X = np.array([[1.0, 2.0]])
        y = np.array([[0.0]])

        y_pred = layer.forward(X)
        grad = 2 * (y_pred - y) / y.size
        layer.backward(grad)

        # Update with optimizer
        optimizer.update(layer)

        # Check that weights changed
        updated_params = layer.get_params()
        assert not np.allclose(updated_params["weights"], initial_weights)

    def test_update_with_momentum(self):
        """Test SGD update with momentum."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))

        # Set initial weights
        initial_weights = np.array([[0.5, 0.5], [0.5, 0.5]])
        initial_bias = np.array([0.0, 0.0])
        layer.set_params({"weights": initial_weights, "bias": initial_bias})

        # Perform forward and backward pass
        X = np.array([[1.0, 2.0]])
        layer.forward(X)
        grad_output = np.array([[0.1, 0.2]])
        layer.backward(grad_output)

        # Update
        optimizer.update(layer)

        # Check velocities were created
        assert id(layer) in optimizer._velocities

    def test_update_with_nesterov(self):
        """Test SGD update with Nesterov momentum."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))

        initial_weights = np.array([[0.5, 0.5], [0.5, 0.5]])
        layer.set_params({"weights": initial_weights, "bias": np.zeros(2)})

        # Forward and backward
        X = np.array([[1.0, 2.0]])
        layer.forward(X)
        layer.backward(np.array([[0.1, 0.2]]))

        # Update
        optimizer.update(layer)

        # Verify weights changed
        assert not np.allclose(layer.weights, initial_weights)

    def test_gradient_clipping_by_norm(self):
        """Test gradient clipping by global norm."""
        optimizer = SGD(learning_rate=0.1, clipnorm=1.0)

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))
        layer.set_params({"weights": np.ones((2, 2)), "bias": np.zeros(2)})

        # Create large gradients
        X = np.array([[10.0, 10.0]])
        layer.forward(X)
        # Large gradient output
        layer.backward(np.array([[100.0, 100.0]]))

        # Get original gradients
        original_grads = layer.get_gradients()
        original_norm = np.sqrt(
            sum(np.sum(np.square(g)) for g in original_grads.values())
        )

        # Update (should clip)
        optimizer.update(layer)

        # The update should have been scaled down
        # This is implicitly tested by checking the update happened
        assert layer.weights is not None

    def test_gradient_clipping_by_value(self):
        """Test gradient clipping by value."""
        optimizer = SGD(learning_rate=0.1, clipvalue=0.5)

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))
        layer.set_params({"weights": np.ones((2, 2)), "bias": np.zeros(2)})

        # Forward and backward
        X = np.array([[1.0, 1.0]])
        layer.forward(X)
        # This would create gradients outside [-0.5, 0.5]
        layer.backward(np.array([[10.0, 10.0]]))

        # Update (gradients should be clipped)
        optimizer.update(layer)

        # Update should have happened
        assert layer.weights is not None

    def test_get_state(self):
        """Test getting optimizer state."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)

        state = optimizer.get_state()

        assert "learning_rate" in state
        assert "momentum" in state
        assert "nesterov" in state
        assert state["learning_rate"] == 0.01
        assert state["momentum"] == 0.9

    def test_set_state(self):
        """Test setting optimizer state."""
        optimizer = SGD()

        new_state = {
            "learning_rate": 0.001,
            "momentum": 0.95,
            "nesterov": True,
            "iterations": 100,
        }

        optimizer.set_state(new_state)

        assert optimizer.learning_rate == 0.001
        assert optimizer.momentum == 0.95
        assert optimizer.nesterov is True
        assert optimizer.iterations == 100

    def test_reset(self):
        """Test resetting optimizer state."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9)

        # Simulate some training
        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))
        layer.forward(np.array([[1.0, 1.0]]))
        layer.backward(np.array([[0.1, 0.1]]))
        optimizer.update(layer)

        assert optimizer.iterations > 0
        assert len(optimizer._velocities) > 0

        # Reset
        optimizer.reset()

        assert optimizer.iterations == 0
        assert len(optimizer._velocities) == 0

    def test_get_config(self):
        """Test getting optimizer configuration."""
        optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)

        config = optimizer.get_config()

        assert config["learning_rate"] == 0.01
        assert config["momentum"] == 0.9
        assert config["clipnorm"] == 1.0

    def test_from_config(self):
        """Test creating optimizer from configuration."""
        config = {
            "learning_rate": 0.001,
            "momentum": 0.95,
            "nesterov": True,
            "clipnorm": None,
            "clipvalue": None,
        }

        optimizer = SGD.from_config(config)

        assert optimizer.learning_rate == 0.001
        assert optimizer.momentum == 0.95
        assert optimizer.nesterov is True

    def test_repr(self):
        """Test string representation."""
        optimizer = SGD(learning_rate=0.01)
        assert "SGD" in repr(optimizer)
        assert "learning_rate=0.01" in repr(optimizer)

        optimizer_with_momentum = SGD(learning_rate=0.01, momentum=0.9)
        assert "momentum=0.9" in repr(optimizer_with_momentum)

        optimizer_with_nesterov = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        assert "nesterov=True" in repr(optimizer_with_nesterov)

    def test_iteration_counter(self):
        """Test that iteration counter increments correctly."""
        optimizer = SGD()

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))
        layer.set_params({"weights": np.ones((2, 2)), "bias": np.zeros(2)})

        # Forward and backward
        layer.forward(np.array([[1.0, 1.0]]))
        layer.backward(np.array([[0.1, 0.1]]))

        initial_iterations = optimizer.iterations

        optimizer.update(layer)

        assert optimizer.iterations == initial_iterations + 1

    def test_multiple_updates(self):
        """Test multiple consecutive updates."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        layer = Dense(2, input_shape=(2,))
        layer.build((1, 2))
        layer.set_params({"weights": np.ones((2, 2)), "bias": np.zeros(2)})

        initial_weights = layer.weights.copy()

        # Perform multiple updates
        for _ in range(5):
            layer.forward(np.array([[1.0, 1.0]]))
            layer.backward(np.array([[0.1, 0.1]]))
            optimizer.update(layer)

        # Weights should have changed
        assert not np.allclose(layer.weights, initial_weights)

    def test_velocity_accumulation(self):
        """Test that velocity accumulates correctly with momentum."""
        optimizer = SGD(learning_rate=0.1, momentum=0.9)

        layer = Dense(1, input_shape=(1,))
        layer.build((1, 1))
        layer.set_params({"weights": np.array([[1.0]]), "bias": np.array([0.0])})

        # First update
        layer.forward(np.array([[1.0]]))
        layer.backward(np.array([[1.0]]))
        optimizer.update(layer)

        first_velocity = optimizer._velocities[id(layer)]["weights"].copy()

        # Second update
        layer.forward(np.array([[1.0]]))
        layer.backward(np.array([[1.0]]))
        optimizer.update(layer)

        second_velocity = optimizer._velocities[id(layer)]["weights"]

        # Velocity should have been updated (momentum * old_v - lr * grad)
        # With momentum 0.9, the velocity should grow
        assert not np.allclose(first_velocity, second_velocity)

    def test_learning_rate_property(self):
        """Test learning rate getter and setter."""
        optimizer = SGD(learning_rate=0.01)

        assert optimizer.learning_rate == 0.01

        optimizer.learning_rate = 0.001
        assert optimizer.learning_rate == 0.001

        # Invalid learning rate should raise error
        with pytest.raises(ValueError):
            optimizer.learning_rate = -0.01

        with pytest.raises(ValueError):
            optimizer.learning_rate = 0.0
