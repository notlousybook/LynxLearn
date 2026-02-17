"""
Linear models for regression tasks.
"""

from ._ols import LinearRegression
from ._gradient import GradientDescentRegressor
from ._ridge import Ridge
from ._lasso import Lasso
from ._elasticnet import ElasticNet
from ._polynomial import PolynomialRegression, PolynomialFeatures
from ._huber import HuberRegressor
from ._quantile import QuantileRegressor
from ._bayesian import BayesianRidge
from ._sgd import SGDRegressor
from ._lars import Lars, LassoLars
from ._omp import OrthogonalMatchingPursuit
from ._ransac import RANSACRegressor
from ._theilsen import TheilSenRegressor
from ._ard import ARDRegression
from ._weighted import WeightedLeastSquares, GeneralizedLeastSquares
from ._glm import (
    GeneralizedLinearModel,
    PoissonRegressor,
    GammaRegressor,
    TweedieRegressor,
)
from ._svr import LinearSVR
from ._isotonic import IsotonicRegression
from ._multitask import MultiTaskElasticNet, MultiTaskLasso

__all__ = [
    "LinearRegression",
    "GradientDescentRegressor",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "PolynomialRegression",
    "PolynomialFeatures",
    "HuberRegressor",
    "QuantileRegressor",
    "BayesianRidge",
    "SGDRegressor",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "RANSACRegressor",
    "TheilSenRegressor",
    "ARDRegression",
    "WeightedLeastSquares",
    "GeneralizedLeastSquares",
    "GeneralizedLinearModel",
    "PoissonRegressor",
    "GammaRegressor",
    "TweedieRegressor",
    "LinearSVR",
    "IsotonicRegression",
    "MultiTaskElasticNet",
    "MultiTaskLasso",
]