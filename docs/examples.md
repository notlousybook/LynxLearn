# Examples

## Quick Start - Linear Models

```python
import numpy as np
from lynxlearn.linear_model import LinearRegression
from lynxlearn.model_selection import train_test_split
from lynxlearn import metrics

X = np.random.randn(100, 1)
y = 3 * X.flatten() + 5 + np.random.randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.train(X_train, y_train)

predictions = model.predict(X_test)

print(f"R² Score: {metrics.r2_score(y_test, predictions):.4f}")
print(f"RMSE: {metrics.rmse(y_test, predictions):.4f}")
```

---

## Quick Start - Neural Networks

### Basic Neural Network

```python
import numpy as np
from lynxlearn import Sequential, Dense, SGD

# Generate data
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.sum(X[:, :3] ** 2, axis=1, keepdims=True) + np.random.randn(1000, 1) * 0.1

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile and train
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse')
history = model.train(X, y, epochs=100, batch_size=32, verbose=1)

# Predict
predictions = model.predict(X[:5])
print(predictions)
```

### Binary Classification (XOR)

```python
import numpy as np
from lynxlearn import Sequential, Dense, SGD

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Build model (needs hidden layer for XOR)
model = Sequential([
    Dense(8, activation='tanh', input_shape=(2,)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=SGD(learning_rate=0.5, momentum=0.9), loss='mse')
model.train(X, y, epochs=500, batch_size=4, verbose=0)

# Predictions
predictions = model.predict(X)
classes = (predictions > 0.5).astype(int)
print(f"Predictions: {classes.flatten()}")
print(f"Actual: {y.flatten()}")
```

### Multi-class Classification

```python
import numpy as np
from lynxlearn import Sequential, Dense, SGD

# Generate 3-class data
np.random.seed(42)
n_samples = 300
X = np.random.randn(n_samples, 5)

# Create one-hot encoded labels
y_indices = np.random.randint(0, 3, n_samples)
y = np.zeros((n_samples, 3))
y[np.arange(n_samples), y_indices] = 1

# Build classifier
model = Sequential([
    Dense(32, activation='relu', input_shape=(5,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=0.1, momentum=0.9), loss='mse')
model.train(X, y, epochs=50, batch_size=16, verbose=0)

# Predict classes
predictions = model.predict(X[:10])
classes = np.argmax(predictions, axis=1)
print(f"Predicted classes: {classes}")
```

---

## Custom Precision

### Float16 (Half Precision)

```python
from lynxlearn import Sequential, DenseFloat16, SGD

model = Sequential([
    DenseFloat16(128, activation='relu', input_shape=(10,)),
    DenseFloat16(64, activation='relu'),
    DenseFloat16(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

### Float64 (Double Precision)

```python
from lynxlearn import Sequential, DenseFloat64, SGD

model = Sequential([
    DenseFloat64(128, activation='relu', input_shape=(10,)),
    DenseFloat64(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

### BF16 (Brain Float 16)

```python
# Requires: pip install ml-dtypes
from lynxlearn import Sequential, DenseBF16, SGD

model = Sequential([
    DenseBF16(128, activation='relu', input_shape=(10,)),
    DenseBF16(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

### Mixed Precision

```python
from lynxlearn import Sequential, DenseMixedPrecision, SGD

model = Sequential([
    DenseMixedPrecision(128, activation='relu', 
                        storage_dtype='float16', 
                        compute_dtype='float32',
                        input_shape=(10,)),
    DenseMixedPrecision(1, storage_dtype='float16', compute_dtype='float32')
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

---

## With Regularization

### L2 Regularization (Ridge)

```python
from lynxlearn import Sequential, Dense, L2Regularizer, SGD

model = Sequential([
    Dense(128, activation='relu',
          kernel_regularizer=L2Regularizer(l2=0.01),
          input_shape=(10,)),
    Dense(64, activation='relu',
          kernel_regularizer=L2Regularizer(l2=0.01)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse')
```

### L1 Regularization (Lasso)

```python
from lynxlearn import Sequential, Dense, L1Regularizer, SGD

model = Sequential([
    Dense(128, activation='relu',
          kernel_regularizer=L1Regularizer(l1=0.01),
          input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

### Elastic Net (L1 + L2)

```python
from lynxlearn import Sequential, Dense, L1L2Regularizer, SGD

model = Sequential([
    Dense(128, activation='relu',
          kernel_regularizer=L1L2Regularizer(l1=0.01, l2=0.01),
          input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

---

## With Constraints

### MaxNorm Constraint

```python
from lynxlearn import Sequential, Dense, MaxNorm, SGD

model = Sequential([
    Dense(128, activation='relu',
          kernel_constraint=MaxNorm(max_value=3.0),
          input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

### Non-Negative Weights

```python
from lynxlearn import Sequential, Dense, NonNeg, SGD

model = Sequential([
    Dense(128, activation='relu',
          kernel_constraint=NonNeg(),
          input_shape=(10,)),
    Dense(1, kernel_constraint=NonNeg())
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
```

---

## Custom Activations

### Using String Identifiers

```python
from lynxlearn import Dense

# All available activations
layer_relu = Dense(64, activation='relu')
layer_sigmoid = Dense(64, activation='sigmoid')
layer_tanh = Dense(64, activation='tanh')
layer_gelu = Dense(64, activation='gelu')
layer_swish = Dense(64, activation='swish')
layer_mish = Dense(64, activation='mish')
layer_leaky = Dense(64, activation='leaky_relu')
layer_leaky_custom = Dense(64, activation='leaky_relu_0.2')  # custom alpha
```

### Custom Activation Function

```python
from lynxlearn import Dense, ActivationRegistry

# Define custom activation
def my_swish_forward(z):
    return z / (1 + np.exp(-z))

def my_swish_backward(grad, z, output):
    sig = 1 / (1 + np.exp(-z))
    return grad * (sig + z * sig * (1 - sig))

# Register it
ActivationRegistry.register('my_swish', my_swish_forward, my_swish_backward)

# Use it
layer = Dense(64, activation='my_swish')
```

---

## Custom Initializers

### Using String Identifiers

```python
from lynxlearn import Dense

# For ReLU-family activations
layer = Dense(64, activation='relu', kernel_initializer='he_normal')
layer = Dense(64, activation='relu', kernel_initializer='he_uniform')

# For tanh/sigmoid
layer = Dense(64, activation='tanh', kernel_initializer='xavier_normal')
layer = Dense(64, activation='sigmoid', kernel_initializer='xavier_uniform')

# For SELU
layer = Dense(64, activation='selu', kernel_initializer='lecun_normal')
```

### Custom Initializer

```python
import numpy as np
from lynxlearn.neural_network.initializers import BaseInitializer

class MyInitializer(BaseInitializer):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale
    
    def initialize(self, shape):
        return np.random.randn(*shape) * self.scale

# Use it
from lynxlearn import Dense
layer = Dense(64, kernel_initializer=MyInitializer(scale=0.05))
```

---

## Training Features

### With Validation Split

```python
from lynxlearn import Sequential, Dense, SGD

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')

# 20% of training data used for validation
history = model.train(X_train, y_train, 
                      epochs=100, 
                      validation_split=0.2,
                      verbose=1)

# Access validation loss
print(f"Final val loss: {history['val_loss'][-1]:.4f}")
```

### With Validation Data

```python
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
```

### Gradient Clipping

```python
from lynxlearn import SGD

# Clip by norm
optimizer = SGD(learning_rate=0.01, momentum=0.9, clipnorm=1.0)

# Clip by value
optimizer = SGD(learning_rate=0.01, momentum=0.9, clipvalue=5.0)
```

---

## Model Evaluation

```python
from lynxlearn import Sequential, Dense, SGD, MSE

model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(1)
])

model.compile(optimizer=SGD(learning_rate=0.01), loss='mse')
model.train(X_train, y_train, epochs=100, verbose=0)

# Evaluate
results = model.evaluate(X_test, y_test)
print(f"Test loss: {results['loss']:.4f}")

# Model summary
model.summary()

# Count parameters
print(f"Total params: {model.count_params()}")

# Get/set weights
weights = model.get_weights()
model.set_weights(weights)
```

---

## Demo Scripts

### Linear Models Demo

```bash
python examples/demo.py
```

This will:
1. Generate synthetic data
2. Train all linear model types
3. Compare their performance
4. Generate visualization plots in `examples/output/`

### Neural Network Demo

```bash
python examples/neural_network_demo.py
```

This demonstrates:
1. Simple regression
2. Non-linear regression (sin function)
3. XOR problem (binary classification)
4. Multi-class classification
5. Optimizer comparison
6. Loss function comparison
7. Weight initialization impact

---

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest tests/

# Verbose output
pytest tests/ -v

# Specific test file
pytest tests/test_neural_network/

# Specific test
pytest tests/test_neural_network/test_integration.py::TestSequentialModel::test_simple_regression
```

Or use the simple test runner (no pytest required):

```bash
python tests/run_tests.py
```
