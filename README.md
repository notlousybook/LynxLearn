# LynxLearn

**A beginner-friendly machine learning library built from scratch with NumPy.**

Educational. Transparent. CPU-optimized for small-to-medium models.

**Made by [lousybook01](https://github.com/notlousybook)** | **YouTube: [LousyBook](https://youtube.com/channel/UCBNE8MNvq1XppUmpAs20m4w)**

---

## Why LynxLearn?

### Where We Excel

| Feature | LynxLearn | PyTorch (CPU) | TensorFlow (CPU) |
|---------|-----------|---------------|------------------|
| Neural Network Training | **2-5x faster** | baseline | 2-3x slower |
| Framework Overhead | **Near zero** | High | Very High |
| Code Readability | **Pure NumPy** | C++ backend | Complex graph |
| Beginner Friendly | ✅ Simple API | Moderate | Steep learning curve |
| Educational Value | ✅ Learn ML fundamentals | Abstraction layers | Hidden complexity |

### Honest Performance Claims

**We WIN at:**
- 🚀 **Neural networks on CPU** - 2-5x faster than PyTorch, 3-10x faster than TensorFlow
- 📚 **Educational value** - Every line is readable NumPy, perfect for learning
- 🎯 **Small-to-medium models** - Where framework overhead dominates
- 🔧 **Customization** - Full control over dtypes, initializers, regularizers

**We DON'T claim to beat:**
- ❌ scikit-learn for linear regression (they have decades of optimization)
- ❌ GPU-accelerated frameworks for large models
- ❌ Production systems requiring distributed training

**Our NICHE:** Educational ML library for learning, prototyping, and CPU-based inference.

---

## Features

### Linear Models
- LinearRegression (OLS), GradientDescentRegressor
- Ridge, Lasso, ElasticNet (regularized regression)
- PolynomialRegression, HuberRegressor, QuantileRegressor, BayesianRidge

### Neural Networks
- Sequential model (Keras-like API)
- Dense layers with multiple activations (ReLU, GELU, Swish, Mish, etc.)
- Multiple precision support: float16, float32, float64, bfloat16
- Weight initializers: He, Xavier, LeCun, Orthogonal
- Regularizers: L1, L2, Elastic Net
- Constraints: MaxNorm, NonNeg, UnitNorm

### Model Selection & Metrics
- train_test_split
- MSE, RMSE, MAE, R² score

### Visualizations
- Regression plots, cost history, residual analysis
- Model comparison charts

---

## Installation

```bash
# Basic installation
pip install lynxlearn

# With BF16 support
pip install lynxlearn[bf16]

# With all features
pip install lynxlearn[all]
```

Or install from source:

```bash
git clone https://github.com/notlousybook/LynxLearn.git
cd LynxLearn
pip install -e .
```

---

## Quick Start

### Linear Regression

```python
import numpy as np
from lynxlearn import LinearRegression, train_test_split, metrics

# Generate data
X = np.random.randn(100, 1)
y = 3 * X.flatten() + 5 + np.random.randn(100) * 0.5

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.train(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print(f"R² Score: {metrics.r2_score(y_test, predictions):.4f}")
```

### Neural Network

```python
from lynxlearn import Sequential, Dense, SGD

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile and train
model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='mse')
history = model.train(X_train, y_train, epochs=100, batch_size=32)

# Predict
predictions = model.predict(X_test)
```

### Custom Precision

```python
from lynxlearn import DenseBF16, DenseFloat16, DenseMixedPrecision

# BF16 precision (requires ml-dtypes)
model = Sequential([
    DenseBF16(128, activation='relu', input_shape=(10,)),
    DenseBF16(1)
])

# Mixed precision training
layer = DenseMixedPrecision(128, storage_dtype='float16', compute_dtype='float32')
```

### With Regularization

```python
from lynxlearn import Dense, L2Regularizer, MaxNorm

layer = Dense(
    128, 
    activation='relu',
    kernel_regularizer=L2Regularizer(l2=0.01),
    kernel_constraint=MaxNorm(3.0)
)
```

---

## Performance Benchmarks

### Neural Network Training (CPU)

| Model Size | LynxLearn | PyTorch | TensorFlow | Winner |
|------------|-----------|---------|------------|--------|
| ~1K params | 0.05s | 0.12s | 0.35s | **LynxLearn 2.4x** |
| ~10K params | 0.15s | 0.45s | 1.2s | **LynxLearn 3x** |
| ~100K params | 0.8s | 2.1s | 5.5s | **LynxLearn 2.6x** |

*Fair comparison: same architecture, same data, same training parameters, CPU-only.*

### Why We're Faster on CPU

```
PyTorch/TensorFlow overhead per layer:
├── Autograd tape recording
├── Dynamic graph construction  
├── CUDA availability checks
├── Distributed training hooks
├── Mixed precision handling
└── Safety checks and assertions

LynxLearn overhead per layer:
└── x @ W + b  (single BLAS call)
```

### What We DON'T Beat

| Task | Winner | Why |
|------|--------|-----|
| Linear Regression | scikit-learn | 20+ years of optimization |
| Large models on GPU | PyTorch/TensorFlow | GPU acceleration |
| Distributed training | PyTorch/TensorFlow | Multi-GPU/TPU support |

---

## Documentation

- [API Reference](docs/api.md) - Complete API documentation
- [Examples](docs/examples.md) - Code examples and tutorials
- [Mathematics](docs/mathematics.md) - Mathematical foundations

---

## Project Structure

```
LynxLearn/
├── lynxlearn/
│   ├── linear_model/      # Linear regression models
│   ├── neural_network/    # Neural network components
│   │   ├── layers/        # Dense, regularizers, constraints
│   │   ├── optimizers/    # SGD with momentum
│   │   ├── losses/        # MSE, MAE, Huber
│   │   └── initializers/  # He, Xavier, LeCun
│   ├── model_selection/   # Train/test split
│   ├── metrics/           # Evaluation metrics
│   └── visualizations/    # Plotting utilities
├── tests/                 # Test suite
├── examples/              # Example scripts
├── benchmark/             # Fair benchmarks
└── docs/                  # Documentation
```

---

## Philosophy

### Transparency

We're honest about performance. We don't cherry-pick unfair comparisons.
Our benchmarks compare apples-to-apples: same algorithm, same data, same hardware.

### Educational Value

Every component is built from scratch with NumPy. No black boxes.
Perfect for students, researchers, and anyone who wants to understand ML fundamentals.

### Beginner-Friendly API

```python
# Simple, intuitive method names
model.train(X, y)      # Not fit()
model.predict(X)       # Clear and obvious
model.evaluate(X, y)   # Returns metrics dictionary
model.summary()        # Print model architecture
```

---

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ -v --cov=lynxlearn

# Run neural network tests only
pytest tests/test_neural_network/
```

---

## Running Benchmarks

```bash
# Quick benchmark
python benchmark/benchmark_neural_network.py --quick

# Full benchmark
python benchmark/benchmark_neural_network.py
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License