"""
Model selection utilities.

This module provides utilities for selecting and preparing data for machine learning models.

Available Functions
-------------------
train_test_split : Split arrays into random train and test subsets.

Examples
--------
>>> from lousybookml.model_selection import train_test_split
>>>
>>> # Basic usage
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>>
>>> # With random seed for reproducibility
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=42, shuffle=True
... )
"""

from ._split import train_test_split

__all__ = ["train_test_split"]
