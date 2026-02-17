# API Reference

## Linear Models

### LinearRegression

Ordinary Least Squares linear regression using the closed-form Normal Equation.

```python
from lynxlearn.linear_model import LinearRegression

model = LinearRegression(learn_bias=True)
model.train(X_train, y_train)
predictions = model.predict(X_test)
score = model.evaluate(X_test, y_test)
```

### GradientDescentRegressor

Linear regression using batch gradient descent optimization.

```python
from lynxlearn.linear_model import GradientDescentRegressor

model = GradientDescentRegressor(
    learning_rate=0.01,
    n_iterations=1000,
    tolerance=1e-6,
    learn_bias=True
)
model.train(X_train, y_train)

print(model.cost_history)
print(f"Converged in {model.n_iter_} iterations")
```

### Ridge

Ridge regression with L2 regularization.

```python
from lynxlearn.linear_model import Ridge

model = Ridge(alpha=1.0, learn_bias=True)
model.train(X_train, y_train)
```

### Lasso

Lasso regression with L1 regularization (automatic feature selection).

```python
from lynxlearn.linear_model import Lasso

model = Lasso(alpha=0.1, max_iter=1000)
model.train(X_train, y_train)

print(f"Non-zero weights: {np.sum(model.weights != 0)}")
```

### ElasticNet

ElasticNet with combined L1 and L2 regularization.

```python
from lynxlearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.train(X_train, y_train)
```

### PolynomialRegression

Polynomial regression using basis expansion.

```python
from lynxlearn.linear_model import PolynomialRegression, PolynomialFeatures

model = PolynomialRegression(degree=3)
model.train(X_train, y_train)

poly = PolynomialFeatures(degree=2)
X_poly = poly.prepare_and_transform(X)
```

### HuberRegressor

Robust regression using Huber loss (less sensitive to outliers).

```python
from lynxlearn.linear_model import HuberRegressor

model = HuberRegressor(epsilon=1.35, alpha=0.0)
model.train(X_train, y_train)
```

### QuantileRegressor

Quantile regression for prediction intervals.

```python
from lynxlearn.linear_model import QuantileRegressor

model = QuantileRegressor(quantile=0.5, alpha=0.0)
model.train(X_train, y_train)

model_10 = QuantileRegressor(quantile=0.1)
model_10.train(X_train, y_train)
model_90 = QuantileRegressor(quantile=0.9)
model_90.train(X_train, y_train)
```

### BayesianRidge

Bayesian approach to ridge regression with uncertainty estimates.

```python
from lynxlearn.linear_model import BayesianRidge

model = BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
model.train(X_train, y_train)

predictions, std = model.predict(X_test, return_std=True)
```

## Model Selection

```python
from lynxlearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
```

## Metrics

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

## Visualizations

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

## Beginner-Friendly Method Names

LynxLearn uses simple, easy-to-understand method names:

| Method | Description |
|--------|-------------|
| `model.train(X, y)` | Train the model on your data |
| `model.predict(X)` | Make predictions on new data |
| `model.evaluate(X, y)` | Evaluate how good the model is (returns R² score) |
| `model.summary()` | Print a summary of the trained model |
| `model.get_params()` | Get the learned weights and bias |

### Backward Compatibility

If you're familiar with scikit-learn, these aliases still work:

| Scikit-learn | LynxLearn |
|--------------|-----------|
| `fit()` | `train()` |
| `score()` | `evaluate()` |
| `coef_` | `weights` |
| `intercept_` | `bias` |