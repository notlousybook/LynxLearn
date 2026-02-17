# LynxLearn

A machine learning library, faster than TensorFlow at training (~200x) while maintaining the same accuracy.

**Made by [lousybook01](https://github.com/notlousybook)** | **YouTube: [LousyBook](https://youtube.com/channel/UCBNE8MNvq1XppUmpAs20m4w)**

## Features

- **Linear Models**: LinearRegression, Ridge, Lasso, ElasticNet, PolynomialRegression, HuberRegressor, QuantileRegressor, BayesianRidge
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
model.train(X_train, y_train)

predictions = model.predict(X_test)

print(f"R² Score: {metrics.r2_score(y_test, predictions):.4f}")
```

## Documentation

- [API Reference](docs/api.md)
- [Examples](docs/examples.md)
- [Benchmarks](docs/benchmarks.md)
- [Mathematics](docs/mathematics.md)

## Project Structure

```
LynxLearn/
├── lynxlearn/
│   ├── linear_model/
│   ├── model_selection/
│   ├── metrics/
│   └── visualizations/
├── tests/
├── examples/
├── benchmark/
└── docs/
```

## License

MIT License
