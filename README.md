# LynxLearn

A simple, educational machine learning library built from scratch using NumPy. Designed for learning and understanding the fundamentals of machine learning algorithms.

**Made by [lousybook01](https://github.com/notlousybook)** | **YouTube: [LousyBook](https://youtube.com/channel/UCBNE8MNvq1XppUmpAs20m4w)** | **Emotional support by Ang!**

## Features

### Linear Models
- **LinearRegression**: Ordinary Least Squares using the Normal Equation
- **GradientDescentRegressor**: Batch Gradient Descent optimization
- **Ridge**: L2 Regularized linear regression
- **Lasso**: L1 Regularized linear regression (feature selection)
- **ElasticNet**: Combined L1 + L2 regularization
- **PolynomialRegression**: Polynomial basis expansion + Linear Regression
- **HuberRegressor**: Robust regression with Huber loss
- **QuantileRegressor**: Quantile regression for prediction intervals
- **BayesianRidge**: Bayesian approach to ridge regression

### Utilities
- **Model Selection**: Train/test split functionality
- **Metrics**: MSE, RMSE, MAE, R² score
- **Visualizations**: Comprehensive plotting utilities

## Installation

```bash
pip install lynxlearn
```

Or install from source:

```bash
git clone https://github.com/notlousybook/LynxLearn.git
cd LynxLearn
pip install -e .
```

## Testing

Run the test suite with pytest:

```bash
pytest tests/

pytest tests/ -v

pytest tests/ --cov=lynxlearn --cov-report=html

pytest tests/test_linear_model.py

pytest tests/test_linear_model.py::TestLinearRegression
```

Or use the simple test runner (no pytest required):

```bash
python tests/run_tests.py
```

## Quick Start

```python
import numpy as np
from lynxlearn.linear_model import LinearRegression
from lynxlearn.model_selection import train_test_split
from lynxlearn import metrics

X = np.random.randn(100, 1)
y = 3 * X.flatten() + 5 + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"R² Score: {metrics.r2_score(y_test, predictions):.4f}")
print(f"RMSE: {metrics.rmse(y_test, predictions):.4f}")
```

## API Reference

### Linear Models

#### LinearRegression

Ordinary Least Squares linear regression using the closed-form Normal Equation.

```python
from lynxlearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = model.score(X_test, y_test)
```

#### GradientDescentRegressor

Linear regression using batch gradient descent optimization.

```python
from lynxlearn.linear_model import GradientDescentRegressor

model = GradientDescentRegressor(
    learning_rate=0.01,
    n_iterations=1000,
    tolerance=1e-6,
    fit_intercept=True
)
model.fit(X_train, y_train)

print(model.cost_history)
print(f"Converged in {model.n_iter_} iterations")
```

#### Ridge

Ridge regression with L2 regularization.

```python
from lynxlearn.linear_model import Ridge

model = Ridge(alpha=1.0, fit_intercept=True)
model.fit(X_train, y_train)
```

#### Lasso

Lasso regression with L1 regularization (automatic feature selection).

```python
from lynxlearn.linear_model import Lasso

model = Lasso(alpha=0.1, max_iter=1000)
model.fit(X_train, y_train)

print(f"Non-zero weights: {np.sum(model.weights != 0)}")
```

#### ElasticNet

ElasticNet with combined L1 and L2 regularization.

```python
from lynxlearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)
```

#### PolynomialRegression

Polynomial regression using basis expansion.

```python
from lynxlearn.linear_model import PolynomialRegression, PolynomialFeatures

model = PolynomialRegression(degree=3)
model.fit(X_train, y_train)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

#### HuberRegressor

Robust regression using Huber loss (less sensitive to outliers).

```python
from lynxlearn.linear_model import HuberRegressor

model = HuberRegressor(epsilon=1.35, alpha=0.0)
model.fit(X_train, y_train)
```

#### QuantileRegressor

Quantile regression for prediction intervals.

```python
from lynxlearn.linear_model import QuantileRegressor

model = QuantileRegressor(quantile=0.5, alpha=0.0)
model.fit(X_train, y_train)

model_10 = QuantileRegressor(quantile=0.1).fit(X_train, y_train)
model_90 = QuantileRegressor(quantile=0.9).fit(X_train, y_train)
```

#### BayesianRidge

Bayesian approach to ridge regression with uncertainty estimates.

```python
from lynxlearn.linear_model import BayesianRidge

model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
model.fit(X_train, y_train)

predictions, std = model.predict(X_test, return_std=True)
```

### Model Selection

```python
from lynxlearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
```

### Metrics

```python
from lynxlearn import metrics

mse = metrics.mean_squared_error(y_true, y_pred)
rmse = metrics.root_mean_squared_error(y_true, y_pred)
mae = metrics.mean_absolute_error(y_true, y_pred)
r2 = metrics.r2_score(y_true, y_pred)

mse = metrics.mse(y_true, y_pred)
rmse = metrics.rmse(y_true, y_pred)
mae = metrics.mae(y_true, y_pred)
```

### Visualizations

```python
from lynxlearn import visualizations
import matplotlib.pyplot as plt

fig = visualizations.plot_regression(X, y, model, title="My Regression")
fig = visualizations.plot_cost_history(gd_model)
fig = visualizations.plot_residuals(y_test, y_pred)
fig = visualizations.plot_coefficients(models, feature_names=["X1", "X2"])
fig = visualizations.compare_models(X_test, y_test, models)

fig = visualizations.visualize_1d_regression(X, y, model)
fig = visualizations.visualize_all_metrics(y_test, y_pred)
fig = visualizations.visualize_gradient_descent(X, y, gd_model)
fig = visualizations.visualize_ridge_comparison(X_train, y_train, X_test, y_test)

report = visualizations.create_comprehensive_report(
    X_train, y_train, X_test, y_test,
    {"OLS": lr, "GD": gd, "Ridge": ridge},
    save_dir="./output"
)
```

## Examples

Run the demo script to see all features in action:

```bash
python examples/demo.py
```

This will:
1. Generate synthetic data
2. Train all model types
3. Compare their performance
4. Generate visualization plots in `examples/output/`

## Project Structure

```
LynxLearn/
├── lynxlearn/
│   ├── __init__.py
│   ├── linear_model/
│   │   ├── __init__.py
│   │   ├── _base.py
│   │   ├── _ols.py
│   │   ├── _gradient.py
│   │   ├── _ridge.py
│   │   ├── _lasso.py
│   │   ├── _elasticnet.py
│   │   ├── _polynomial.py
│   │   ├── _huber.py
│   │   ├── _quantile.py
│   │   └── _bayesian.py
│   ├── model_selection/
│   │   ├── __init__.py
│   │   └── _split.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── _regression.py
│   └── visualizations/
│       ├── __init__.py
│       ├── _plots.py
│       ├── _regression_analysis.py
│       ├── _gradient_descent.py
│       ├── _regularization.py
│       └── _model_comparison.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_linear_model.py
│   ├── test_metrics.py
│   ├── test_model_selection.py
│   ├── test_visualizations.py
│   └── run_tests.py
├── examples/
│   └── demo.py
├── benchmark/
│   ├── benchmark_runner.py
│   ├── benchmark_colab.ipynb
│   └── README.md
├── pyproject.toml
├── README.md
└── requirements-dev.txt
```

## Mathematics

### Ordinary Least Squares (Normal Equation)

```
θ = (XᵀX)⁻¹Xᵀy
```

### Gradient Descent

```
repeat until convergence:
    θ := θ - α * ∇J(θ)
```

### Ridge Regression

```
θ = (XᵀX + λI)⁻¹Xᵀy
```

### Lasso Regression

```
minimize: (1/2n) * ||y - Xw||² + α * ||w||₁
```

### ElasticNet

```
minimize: (1/2n) * ||y - Xw||² + α * (l1_ratio * ||w||₁ + 0.5 * (1 - l1_ratio) * ||w||²)
```

## License

MIT License - Feel free to use for educational purposes!

## Contributing

This is an educational project. Feel free to fork and experiment!

## Links

- **GitHub**: https://github.com/notlousybook/LynxLearn
- **YouTube**: https://youtube.com/channel/UCBNE8MNvq1XppUmpAs20m4w
