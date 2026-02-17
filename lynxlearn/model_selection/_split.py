"""
Dataset splitting utilities.
"""

import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Features array.
    y : array-like of shape (n_samples,)
        Target array.
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split.
        Should be between 0.0 and 1.0.
    random_state : int, optional
        Seed for the random number generator for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
        Split arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from lousybookml.model_selection import train_test_split
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([1, 2, 3, 4])
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    n_samples = X.shape[0]

    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_count = int(n_samples * test_size)
    train_count = n_samples - test_count

    train_indices = indices[:train_count]
    test_indices = indices[train_count:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test