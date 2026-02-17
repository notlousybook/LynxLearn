"""
Tests for model_selection module.
"""

import numpy as np
import pytest

from lynxlearn.model_selection import train_test_split


class TestTrainTestSplit:
    """Tests for train_test_split function."""

    def test_basic_split(self):
        """Test basic train/test split."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        assert len(X_train) == 7
        assert len(X_test) == 3
        assert len(y_train) == 7
        assert len(y_test) == 3

    def test_test_size_20_percent(self):
        """Test with 20% test size."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_reproducibility_with_random_state(self):
        """Test that same random_state gives same split."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_different_random_states(self):
        """Test that different random_states give different splits."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        X_train1, X_test1, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train2, X_test2, _, _ = train_test_split(X, y, test_size=0.2, random_state=24)

        # Should be different (with very high probability)
        assert not np.array_equal(X_train1, X_train2)

    def test_shuffle_false(self):
        """Test split without shuffling."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, shuffle=False
        )

        # Without shuffle, should take last 30% as test
        np.testing.assert_array_equal(X_test.flatten(), [8, 9, 10])
        np.testing.assert_array_equal(y_test, [8, 9, 10])

    def test_consistent_indices(self):
        """Test that X and y indices are consistent."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Check that we can reconstruct the original data
        X_reconstructed = np.concatenate([X_train, X_test])
        y_reconstructed = np.concatenate([y_train, y_test])

        # Sort both to compare (since order is shuffled)
        X_sorted_orig = X[np.argsort(X[:, 0])]
        X_sorted_recon = X_reconstructed[np.argsort(X_reconstructed[:, 0])]

        np.testing.assert_array_almost_equal(X_sorted_orig, X_sorted_recon)

    def test_1d_X(self):
        """Test with 1D X array."""
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        assert len(X_train) == 7
        assert len(X_test) == 3
        assert len(y_train) == 7
        assert len(y_test) == 3

    def test_empty_test_size(self):
        """Test with very small test size."""
        X = np.random.randn(10, 2)
        y = np.random.randn(10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        assert len(X_test) == 1
        assert len(X_train) == 9
