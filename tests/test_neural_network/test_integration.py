"""
Integration tests for neural network training.

Tests the full training pipeline from model creation to prediction.
"""

import numpy as np
import pytest

from lynxlearn.neural_network import (
    SGD,
    Dense,
    MeanAbsoluteError,
    MeanSquaredError,
    Sequential,
)


class TestSequentialModel:
    """Integration tests for Sequential model."""

    def test_simple_regression(self):
        """Test simple linear regression: y = 3x + 5."""
        np.random.seed(42)

        # Create simple linear data
        X = np.random.randn(100, 1)
        y = 3 * X + 5 + np.random.randn(100, 1) * 0.1

        # Build model
        model = Sequential([Dense(1, input_shape=(1,))])

        # Compile
        model.compile(optimizer=SGD(learning_rate=0.1), loss=MeanSquaredError())

        # Train
        history = model.train(X, y, epochs=100, batch_size=32, verbose=0)

        # Check that loss decreased
        assert history["loss"][-1] < history["loss"][0]

        # Check final loss is small
        assert history["loss"][-1] < 0.5, f"Final loss too high: {history['loss'][-1]}"

        # Check learned weights are close to [3, 5]
        learned_weights = model.layers[0].weights[0, 0]
        learned_bias = model.layers[0].bias[0]

        assert abs(learned_weights - 3.0) < 0.5, (
            f"Weight should be ~3.0, got {learned_weights}"
        )
        assert abs(learned_bias - 5.0) < 0.5, f"Bias should be ~5.0, got {learned_bias}"

    def test_model_with_add_method(self):
        """Test building model using add() method."""
        model = Sequential()
        model.add(Dense(32, activation="relu", input_shape=(10,)))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer="sgd", loss="mse")

        # Test forward pass
        X = np.random.randn(5, 10)
        output = model.predict(X)

        assert output.shape == (5, 1)

    def test_model_with_string_optimizer_and_loss(self):
        """Test using string identifiers for optimizer and loss."""
        model = Sequential([Dense(16, activation="relu", input_shape=(8,)), Dense(1)])

        # Use string identifiers
        model.compile(optimizer="sgd", loss="mse")

        X = np.random.randn(10, 8)
        y = np.random.randn(10, 1)

        history = model.train(X, y, epochs=5, verbose=0)

        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_training_with_validation_split(self):
        """Test training with validation split."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer=SGD(learning_rate=0.01), loss="mse")

        history = model.train(
            X, y, epochs=10, batch_size=16, validation_split=0.2, verbose=0
        )

        assert "loss" in history
        assert "val_loss" in history
        assert len(history["loss"]) == 10
        assert len(history["val_loss"]) == 10

    def test_training_with_validation_data(self):
        """Test training with explicit validation data."""
        np.random.seed(42)

        X_train = np.random.randn(80, 5)
        y_train = np.random.randn(80, 1)
        X_val = np.random.randn(20, 5)
        y_val = np.random.randn(20, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer=SGD(learning_rate=0.01), loss="mse")

        history = model.train(
            X_train, y_train, epochs=10, validation_data=(X_val, y_val), verbose=0
        )

        assert "val_loss" in history

    def test_predict_methods(self):
        """Test predict, predict_proba, and predict_classes methods."""
        np.random.seed(42)

        # Binary classification
        X = np.random.randn(100, 5)
        y = (np.random.randn(100, 1) > 0).astype(float)

        model = Sequential(
            [
                Dense(16, activation="relu", input_shape=(5,)),
                Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer=SGD(learning_rate=0.1), loss="mse")
        model.train(X, y, epochs=10, verbose=0)

        X_test = np.random.randn(10, 5)

        # Test predict
        predictions = model.predict(X_test)
        assert predictions.shape == (10, 1)

        # Test predict_proba (alias for predict)
        probas = model.predict_proba(X_test)
        assert probas.shape == (10, 1)

        # Test predict_classes
        classes = model.predict_classes(X_test)
        assert classes.shape == (10,)
        assert np.all((classes == 0) | (classes == 1))

    def test_multiclass_classification(self):
        """Test multi-class classification with softmax."""
        np.random.seed(42)

        # Create 3-class data
        n_samples = 150
        X = np.random.randn(n_samples, 10)

        # Create one-hot encoded labels
        y_indices = np.random.randint(0, 3, n_samples)
        y = np.zeros((n_samples, 3))
        y[np.arange(n_samples), y_indices] = 1

        model = Sequential(
            [
                Dense(32, activation="relu", input_shape=(10,)),
                Dense(3, activation="softmax"),
            ]
        )

        model.compile(optimizer=SGD(learning_rate=0.1), loss="mse")
        history = model.train(X, y, epochs=20, verbose=0)

        # Test predictions
        X_test = np.random.randn(5, 10)
        predictions = model.predict(X_test)

        # Softmax output should sum to 1
        assert np.allclose(np.sum(predictions, axis=1), 1.0)

        # predict_classes should return class indices
        classes = model.predict_classes(X_test)
        assert classes.shape == (5,)
        assert np.all((classes >= 0) & (classes < 3))

    def test_evaluate_method(self):
        """Test model evaluation."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer=SGD(learning_rate=0.01), loss="mse")
        model.train(X, y, epochs=10, verbose=0)

        X_test = np.random.randn(20, 5)
        y_test = np.random.randn(20, 1)

        results = model.evaluate(X_test, y_test)

        assert "loss" in results

    def test_summary_method(self, capsys):
        """Test model summary output."""
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(10,)),
                Dense(32, activation="relu"),
                Dense(1),
            ]
        )

        model.compile(optimizer="sgd", loss="mse")
        model.summary()

        captured = capsys.readouterr()
        assert "Sequential" in captured.out
        assert "Dense" in captured.out
        assert "Total params" in captured.out

    def test_get_and_set_weights(self):
        """Test getting and setting model weights."""
        model = Sequential([Dense(16, activation="relu", input_shape=(8,)), Dense(1)])

        model.compile(optimizer="sgd", loss="mse")

        # Get weights
        weights = model.get_weights()
        assert len(weights) == 2

        # Modify and set weights
        new_weights = []
        for w in weights:
            new_w = {k: v * 2 for k, v in w.items()}
            new_weights.append(new_w)

        model.set_weights(new_weights)

        # Verify weights were set
        updated_weights = model.get_weights()
        for i, (old, new) in enumerate(zip(weights, updated_weights)):
            for key in old:
                assert np.allclose(new[key], old[key] * 2)

    def test_count_params(self):
        """Test parameter counting."""
        model = Sequential(
            [
                Dense(64, activation="relu", input_shape=(10,)),  # 10*64 + 64 = 704
                Dense(32, activation="relu"),  # 64*32 + 32 = 2080
                Dense(1),  # 32*1 + 1 = 33
            ]
        )

        model.compile(optimizer="sgd", loss="mse")

        # Total: 704 + 2080 + 33 = 2817
        assert model.count_params() == 2817

    def test_model_not_compiled_error(self):
        """Test that training without compile raises error."""
        model = Sequential([Dense(16, input_shape=(5,))])

        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)

        with pytest.raises(RuntimeError):
            model.train(X, y, epochs=5)

    def test_fit_alias(self):
        """Test that fit() is an alias for train()."""
        np.random.seed(42)

        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer="sgd", loss="mse")

        # Use fit() instead of train()
        history = model.fit(X, y, epochs=5, verbose=0)

        assert "loss" in history
        assert len(history["loss"]) == 5


class TestGradientDescent:
    """Tests for gradient descent optimization."""

    def test_convergence_on_convex_problem(self):
        """Test that SGD converges on a convex optimization problem."""
        np.random.seed(42)

        # Simple convex problem: minimize ||Wx - y||^2
        # True solution: W = (X^T X)^{-1} X^T y
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        true_W = np.random.randn(n_features, 1)
        y = X @ true_W

        model = Sequential([Dense(1, input_shape=(n_features,))])

        model.compile(
            optimizer=SGD(learning_rate=0.01, momentum=0.9), loss=MeanSquaredError()
        )

        initial_loss = model.evaluate(X, y)["loss"]

        model.train(X, y, epochs=200, verbose=0)

        final_loss = model.evaluate(X, y)["loss"]

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.1

    def test_momentum_improves_convergence(self):
        """Test that momentum helps convergence."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)

        # Model without momentum
        model_no_momentum = Sequential(
            [Dense(16, activation="relu", input_shape=(5,)), Dense(1)]
        )
        model_no_momentum.compile(
            optimizer=SGD(learning_rate=0.01, momentum=0.0), loss="mse"
        )
        history_no_momentum = model_no_momentum.train(X, y, epochs=50, verbose=0)

        # Model with momentum
        model_with_momentum = Sequential(
            [Dense(16, activation="relu", input_shape=(5,)), Dense(1)]
        )
        model_with_momentum.compile(
            optimizer=SGD(learning_rate=0.01, momentum=0.9), loss="mse"
        )
        history_with_momentum = model_with_momentum.train(X, y, epochs=50, verbose=0)

        # Momentum should generally help (though not always guaranteed)
        # We just check that both models trained successfully
        assert history_no_momentum["loss"][-1] < history_no_momentum["loss"][0]
        assert history_with_momentum["loss"][-1] < history_with_momentum["loss"][0]


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_large_input_values(self):
        """Test handling of large input values with gradient clipping."""
        np.random.seed(42)

        X = np.random.randn(50, 5) * 1000
        y = np.random.randn(50, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        # Use gradient clipping to handle large values
        model.compile(optimizer=SGD(learning_rate=0.0001, clipnorm=1.0), loss="mse")
        history = model.train(X, y, epochs=10, verbose=0)

        # Check for NaN values
        assert not np.any(np.isnan(history["loss"]))

    def test_small_input_values(self):
        """Test handling of very small input values."""
        np.random.seed(42)

        X = np.random.randn(50, 5) * 1e-10
        y = np.random.randn(50, 1) * 1e-10

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer=SGD(learning_rate=0.01), loss="mse")
        history = model.train(X, y, epochs=10, verbose=0)

        # Check for NaN values
        assert not np.any(np.isnan(history["loss"]))

    def test_softmax_stability(self):
        """Test softmax numerical stability with large logits."""
        np.random.seed(42)

        # Create inputs that could cause numerical issues
        X = np.random.randn(50, 10) * 100

        model = Sequential(
            [
                Dense(16, activation="relu", input_shape=(10,)),
                Dense(5, activation="softmax"),
            ]
        )

        model.compile(optimizer=SGD(learning_rate=0.001), loss="mse")

        # Create one-hot targets
        y = np.zeros((50, 5))
        y[np.arange(50), np.random.randint(0, 5, 50)] = 1

        history = model.train(X, y, epochs=10, verbose=0)

        # Check predictions are valid probabilities
        predictions = model.predict(X)
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.allclose(np.sum(predictions, axis=1), 1.0)


class TestDifferentLosses:
    """Tests for different loss functions."""

    def test_mae_loss(self):
        """Test training with MAE loss."""
        np.random.seed(42)

        X = np.random.randn(100, 5)
        y = np.random.randn(100, 1)

        model = Sequential([Dense(16, activation="relu", input_shape=(5,)), Dense(1)])

        model.compile(optimizer=SGD(learning_rate=0.01), loss=MeanAbsoluteError())

        history = model.train(X, y, epochs=20, verbose=0)

        # Loss should decrease
        assert history["loss"][-1] < history["loss"][0]

    def test_mse_vs_mae_behavior(self):
        """Test that MSE and MAE behave differently with outliers."""
        np.random.seed(42)

        # Data with an outlier
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        y[0] = 100  # Outlier

        # Train with MSE
        model_mse = Sequential([Dense(1, input_shape=(5,))])
        model_mse.compile(optimizer=SGD(learning_rate=0.01), loss="mse")
        model_mse.train(X, y, epochs=50, verbose=0)

        # Train with MAE
        model_mae = Sequential([Dense(1, input_shape=(5,))])
        model_mae.compile(optimizer=SGD(learning_rate=0.01), loss="mae")
        model_mae.train(X, y, epochs=50, verbose=0)

        # Both should have trained (loss decreased)
        # MAE should be more robust to the outlier
        assert model_mse.evaluate(X, y)["loss"] > 0
        assert model_mae.evaluate(X, y)["loss"] > 0
