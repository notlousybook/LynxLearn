# API Reference

## Table of Contents

- [Linear Models](#linear-models)
- [Neural Network Models](#neural-network-models)
- [Layers](#layers)
- [Optimizers](#optimizers)
- [Loss Functions](#loss-functions)
- [Regularizers](#regularizers)
- [Constraints](#constraints)
- [Weight Initializers](#weight-initializers)
- [Model Selection](#model-selection)
- [Metrics](#metrics)
- [Visualizations](#visualizations)

---

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

---

## Neural Network Models

### Sequential

Linear stack of layers for building neural networks.

```python
from lynxlearn import Sequential, Dense

# Basic model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile with optimizer and loss
model.compile(optimizer='sgd', loss='mse')

# Train with validation
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Predict
predictions = model.predict(X_test)
```

**Compile Options:**
```python
from lynxlearn import SGD, MeanSquaredError

model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss=MeanSquaredError(),
    metrics=['accuracy']
)
```

**Training Options:**
```python
history = model.train(
    X_train, y_train,
    epochs=100,              # Number of training epochs
    batch_size=32,           # Samples per gradient update
    validation_data=(X_val, y_val),  # Validation data
    validation_split=0.2,    # Or use fraction of training data
    shuffle=True,            # Shuffle data each epoch
    verbose=1                # 0=silent, 1=progress, 2=per-epoch
)
```

**Model Methods:**
```python
# Evaluation
results = model.evaluate(X_test, y_test)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Class probabilities
y_classes = model.predict_classes(X_test)  # Class labels

# Summary
model.summary()

# Parameters
total_params = model.count_params()
weights = model.get_weights()
model.set_weights(weights)

# History
loss_history = history['loss']
val_loss_history = history.get('val_loss', [])
```

---

## Layers

### Dense

Universal fully connected layer with full customization.

```python
from lynxlearn import Dense

# Basic usage
layer = Dense(128, activation='relu')

# With all options
layer = Dense(
    units=256,
    activation='gelu',
    use_bias=True,
    kernel_initializer='he_normal',
    bias_initializer='zeros',
    kernel_regularizer={'l2': 0.01},
    kernel_constraint=MaxNorm(3.0),
    dtype='float32'
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `units` | int | required | Number of neurons |
| `activation` | str/callable | None | Activation function |
| `use_bias` | bool | True | Include bias term |
| `kernel_initializer` | str | 'he_normal' | Weight initializer |
| `bias_initializer` | str | 'zeros' | Bias initializer |
| `kernel_regularizer` | Regularizer | None | Weight regularizer |
| `bias_regularizer` | Regularizer | None | Bias regularizer |
| `activity_regularizer` | Regularizer | None | Output regularizer |
| `kernel_constraint` | Constraint | None | Weight constraint |
| `bias_constraint` | Constraint | None | Bias constraint |
| `dtype` | str | 'float32' | Data type |

**Supported Dtypes:**
- `'float16'` / `'half'` - 16-bit floating point
- `'float32'` / `'single'` - 32-bit floating point (default)
- `'float64'` / `'double'` - 64-bit floating point
- `'bfloat16'` / `'bf16'` - Brain float 16 (requires `pip install ml-dtypes`)

**Supported Activations:**
- `'relu'` - Rectified Linear Unit
- `'leaky_relu'` - Leaky ReLU (alpha=0.01)
- `'leaky_relu_0.2'` - Leaky ReLU with custom alpha
- `'sigmoid'` - Sigmoid function
- `'tanh'` - Hyperbolic tangent
- `'softmax'` - Softmax function
- `'elu'` - Exponential Linear Unit
- `'selu'` - Scaled ELU
- `'swish'` / `'silu'` - Swish/SiLU
- `'gelu'` - Gaussian Error Linear Unit
- `'softplus'` - Softplus function
- `'softsign'` - Softsign function
- `'mish'` - Mish activation
- `'linear'` / `None` - No activation

**Convenience Layer Aliases:**
```python
from lynxlearn import DenseFloat16, DenseFloat32, DenseFloat64, DenseBF16

# Dtype-specific layers
layer_f16 = DenseFloat16(128, activation='relu')
layer_f32 = DenseFloat32(128, activation='relu')
layer_f64 = DenseFloat64(128, activation='relu')
layer_bf16 = DenseBF16(128, activation='relu')  # requires ml-dtypes
```

**Mixed Precision:**
```python
from lynxlearn import DenseMixedPrecision

# Store in float16, compute in float32
layer = DenseMixedPrecision(
    units=128,
    storage_dtype='float16',
    compute_dtype='float32',
    activation='relu'
)
```

---

## Optimizers

### SGD

Stochastic Gradient Descent with momentum and gradient clipping support.

```python
from lynxlearn import SGD

# Vanilla SGD
optimizer = SGD(learning_rate=0.01)

# With momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9)

# With Nesterov momentum
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# With gradient clipping
optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)
optimizer = SGD(learning_rate=0.01, momentum=0.9, clipvalue=5.0)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 0.01 | Step size for updates |
| `momentum` | float | 0.0 | Momentum coefficient (0-1) |
| `nesterov` | bool | False | Use Nesterov momentum |
| `clipnorm` | float | None | Clip gradients by global norm |
| `clipvalue` | float | None | Clip gradients by value |

**State Management:**
```python
# Get/set state for checkpointing
state = optimizer.get_state()
optimizer.set_state(state)

# Reset optimizer
optimizer.reset()

# Configuration
config = optimizer.get_config()
optimizer = SGD.from_config(config)
```

---

## Loss Functions

### MeanSquaredError (MSE)

```python
from lynxlearn import MeanSquaredError, MSE

loss = MeanSquaredError(reduction='mean')
loss_value = loss.compute(y_true, y_pred)
gradient = loss.gradient(y_true, y_pred)

# Or compute both at once
loss_value, gradient = loss(y_true, y_pred, return_grad=True)
```

### MeanAbsoluteError (MAE)

```python
from lynxlearn import MeanAbsoluteError, MAE

loss = MeanAbsoluteError(reduction='mean')
```

### HuberLoss

Robust loss combining MSE and MAE, less sensitive to outliers.

```python
from lynxlearn import HuberLoss

loss = HuberLoss(delta=1.0, reduction='mean')
```

**Parameters:**
- `delta` - Threshold for switching between MSE and MAE
- `reduction` - 'mean', 'sum', or 'none'

---

## Regularizers

### L1Regularizer

```python
from lynxlearn import L1Regularizer

layer = Dense(128, kernel_regularizer=L1Regularizer(l1=0.01))
```

### L2Regularizer

```python
from lynxlearn import L2Regularizer

layer = Dense(128, kernel_regularizer=L2Regularizer(l2=0.01))
```

### L1L2Regularizer (Elastic Net)

```python
from lynxlearn import L1L2Regularizer

layer = Dense(128, kernel_regularizer=L1L2Regularizer(l1=0.01, l2=0.01))
```

**Dict Shorthand:**
```python
# You can also use dict shorthand
layer = Dense(128, kernel_regularizer={'l1': 0.01, 'l2': 0.01})
layer = Dense(128, kernel_regularizer='l2')  # Default l2=0.01
```

---

## Constraints

### MaxNorm

Constrain weights to have max norm.

```python
from lynxlearn import MaxNorm

layer = Dense(128, kernel_constraint=MaxNorm(max_value=3.0))
```

### NonNeg

Constrain weights to be non-negative.

```python
from lynxlearn import NonNeg

layer = Dense(128, kernel_constraint=NonNeg())
```

### UnitNorm

Constrain weights to have unit norm.

```python
from lynxlearn import UnitNorm

layer = Dense(128, kernel_constraint=UnitNorm(axis=0))
```

### MinMaxNorm

Constrain weights to have norm within range.

```python
from lynxlearn import MinMaxNorm

layer = Dense(128, kernel_constraint=MinMaxNorm(min_value=0.5, max_value=2.0))
```

---

## Weight Initializers

### He Initializers (for ReLU-family activations)

```python
from lynxlearn import HeNormal, HeUniform

# He normal initialization
layer = Dense(128, activation='relu', kernel_initializer='he_normal')
layer = Dense(128, activation='relu', kernel_initializer=HeNormal())

# He uniform initialization
layer = Dense(128, kernel_initializer='he_uniform')
```

### Xavier/Glorot Initializers (for tanh/sigmoid)

```python
from lynxlearn import XavierNormal, XavierUniform, GlorotNormal, GlorotUniform

layer = Dense(128, activation='tanh', kernel_initializer='xavier_normal')
layer = Dense(128, activation='sigmoid', kernel_initializer=XavierUniform())
```

### LeCun Initializers (for SELU)

```python
from lynxlearn import LeCunNormal, LeCunUniform

layer = Dense(128, activation='selu', kernel_initializer='lecun_normal')
```

### Other Initializers

```python
from lynxlearn import RandomNormal, RandomUniform, Zeros, Ones, Constant, Orthogonal, TruncatedNormal

# Random normal/uniform
layer = Dense(128, kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))
layer = Dense(128, kernel_initializer=RandomUniform(minval=-0.05, maxval=0.05))

# Constant initializers
layer = Dense(128, kernel_initializer=Zeros())
layer = Dense(128, kernel_initializer=Ones())
layer = Dense(128, kernel_initializer=Constant(value=0.5))

# Orthogonal initialization
layer = Dense(128, kernel_initializer=Orthogonal(gain=1.0))

# Truncated normal
layer = Dense(128, kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.05))
```

### Custom Initializers

```python
from lynxlearn.neural_network.initializers import BaseInitializer

class MyInitializer(BaseInitializer):
    def initialize(self, shape):
        return np.random.randn(*shape) * 0.01

layer = Dense(128, kernel_initializer=MyInitializer())
```

---

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

---

## Metrics

```python
from lynxlearn import metrics

mse = metrics.mean_squared_error(y_true, y_pred)
rmse = metrics.root_mean_squared_error(y_true, y_pred)
mae = metrics.mean_absolute_error(y_true, y_pred)
r2 = metrics.r2_score(y_true, y_pred)

# Short aliases
mse = metrics.mse(y_true, y_pred)
rmse = metrics.rmse(y_true, y_pred)
mae = metrics.mae(y_true, y_pred)
```

---

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

---

## Activation Registry

Register custom activation functions:

```python
from lynxlearn import ActivationRegistry

# List available activations
print(ActivationRegistry.list_available())

# Register custom activation
def my_forward(z):
    return z * (z > 0)

def my_backward(grad, z, output):
    return grad * (z > 0)

ActivationRegistry.register('my_relu', my_forward, my_backward)

# Use custom activation
layer = Dense(128, activation='my_relu')
```

---

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

---

## Performance Tips

### Choosing Data Types

| Scenario | Recommended dtype |
|----------|-------------------|
| CPU training (default) | `float32` |
| Maximum precision | `float64` |
| Memory-constrained | `float16` or `bfloat16` |
| Mixed precision | `DenseMixedPrecision` |

### Choosing Initializers

| Activation | Recommended Initializer |
|------------|------------------------|
| ReLU, LeakyReLU, ELU, Swish, GELU | `he_normal` |
| tanh, sigmoid | `xavier_normal` |
| SELU | `lecun_normal` |

### Choosing Optimizers

| Scenario | Recommended Settings |
|----------|---------------------|
| Fast convergence | `SGD(learning_rate=0.01, momentum=0.9)` |
| Stable training | `SGD(learning_rate=0.001, momentum=0.9)` |
| With gradient explosion | `SGD(learning_rate=0.01, clipnorm=1.0)` |