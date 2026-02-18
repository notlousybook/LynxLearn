"""
Tests for neural network layers.
"""

import numpy as np
import pytest

from lynxlearn.neural_network.initializers import HeNormal, XavierNormal, Zeros
from lynxlearn.neural_network.layers import BaseLayer, Dense


class TestDense:
    """Tests for Dense layer."""

    def test_initialization(self):
        """Test Dense layer initialization."""
        layer = Dense(64)

        assert layer.units == 64
        assert layer.activation is None
        assert layer.use_bias is True
        assert layer.built is False

    def test_initialization_with_activation(self):
        """Test Dense layer with activation."""
        layer = Dense(128, activation="relu")

        assert layer.units == 128
        assert layer.activation == "relu"

    def test_initialization_with_parameters(self):
        """Test Dense layer with custom parameters."""
        layer = Dense(
            units=32,
            activation="sigmoid",
            use_bias=False,
            kernel_initializer="xavier",
            bias_initializer="zeros",
        )

        assert layer.units == 32
        assert layer.activation == "sigmoid"
        assert layer.use_bias is False

    def test_invalid_units(self):
        """Test that invalid units raises error."""
        with pytest.raises(ValueError):
            Dense(0)

        with pytest.raises(ValueError):
            Dense(-1)

    def test_build(self):
        """Test layer build."""
        layer = Dense(64, input_shape=(None, 128))
        layer.build((32, 128))

        assert layer.built is True
        assert layer.weights.shape == (128, 64)
        assert layer.bias.shape == (64,)

    def test_build_without_bias(self):
        """Test layer build without bias."""
        layer = Dense(64, use_bias=False)
        layer.build((32, 128))

        assert layer.built is True
        assert layer.bias is None

    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = Dense(32, input_shape=(None, 16))
        layer.build((8, 16))

        X = np.random.randn(8, 16)
        output = layer.forward(X)

        assert output.shape == (8, 32)

    def test_forward_pass_with_relu(self):
        """Test forward pass with ReLU activation."""
        layer = Dense(16, activation="relu")
        layer.build((4, 8))

        X = np.random.randn(4, 8)
        output = layer.forward(X)

        # ReLU output should be non-negative
        assert np.all(output >= 0)

    def test_forward_pass_with_sigmoid(self):
        """Test forward pass with sigmoid activation."""
        layer = Dense(8, activation="sigmoid")
        layer.build((4, 16))

        X = np.random.randn(4, 16)
        output = layer.forward(X)

        # Sigmoid output should be between 0 and 1
        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_forward_pass_with_tanh(self):
        """Test forward pass with tanh activation."""
        layer = Dense(8, activation="tanh")
        layer.build((4, 16))

        X = np.random.randn(4, 16)
        output = layer.forward(X)

        # Tanh output should be between -1 and 1
        assert np.all(output >= -1)
        assert np.all(output <= 1)

    def test_forward_pass_with_softmax(self):
        """Test forward pass with softmax activation."""
        layer = Dense(10, activation="softmax")
        layer.build((4, 16))

        X = np.random.randn(4, 16)
        output = layer.forward(X)

        # Softmax output should sum to 1 for each sample
        assert np.allclose(np.sum(output, axis=1), 1.0)
        assert np.all(output >= 0)

    def test_backward_pass(self):
        """Test backward pass through layer."""
        layer = Dense(32)
        layer.build((8, 16))

        X = np.random.randn(8, 16)
        output = layer.forward(X)

        grad_output = np.random.randn(8, 32)
        grad_input = layer.backward(grad_output)

        assert grad_input.shape == (8, 16)
        assert layer.grad_weights.shape == (16, 32)
        assert layer.grad_bias.shape == (32,)

    def test_backward_without_forward_raises(self):
        """Test that backward without forward raises error."""
        layer = Dense(32)
        layer.build((8, 16))

        grad_output = np.random.randn(8, 32)

        with pytest.raises(RuntimeError):
            layer.backward(grad_output)

    def test_get_params(self):
        """Test get_params method."""
        layer = Dense(32)
        layer.build((8, 16))

        params = layer.get_params()

        assert "weights" in params
        assert "bias" in params
        assert params["weights"].shape == (16, 32)
        assert params["bias"].shape == (32,)

    def test_get_params_without_bias(self):
        """Test get_params without bias."""
        layer = Dense(32, use_bias=False)
        layer.build((8, 16))

        params = layer.get_params()

        assert "weights" in params
        assert "bias" not in params

    def test_set_params(self):
        """Test set_params method."""
        layer = Dense(32)
        layer.build((8, 16))

        new_weights = np.random.randn(16, 32)
        new_bias = np.random.randn(32)

        layer.set_params({"weights": new_weights, "bias": new_bias})

        assert np.allclose(layer.weights, new_weights)
        assert np.allclose(layer.bias, new_bias)

    def test_get_gradients(self):
        """Test get_gradients method."""
        layer = Dense(32)
        layer.build((8, 16))

        X = np.random.randn(8, 16)
        layer.forward(X)
        layer.backward(np.random.randn(8, 32))

        grads = layer.get_gradients()

        assert "weights" in grads
        assert "bias" in grads
        assert grads["weights"].shape == (16, 32)
        assert grads["bias"].shape == (32,)

    def test_count_params(self):
        """Test count_params method."""
        layer = Dense(32)
        layer.build((8, 16))

        # Weights: 16 * 32 = 512, Bias: 32
        assert layer.count_params() == 544

    def test_count_params_without_bias(self):
        """Test count_params without bias."""
        layer = Dense(32, use_bias=False)
        layer.build((8, 16))

        assert layer.count_params() == 512

    def test_count_params_not_built(self):
        """Test count_params when not built."""
        layer = Dense(32)

        assert layer.count_params() == 0

    def test_repr(self):
        """Test string representation."""
        layer = Dense(64, activation="relu")

        assert "Dense" in repr(layer)
        assert "64" in repr(layer)
        assert "relu" in repr(layer)


class TestDenseActivations:
    """Tests for Dense layer activation functions."""

    def test_relu_activation(self):
        """Test ReLU activation."""
        layer = Dense(8, activation="relu")
        layer.build((2, 4))
        layer
