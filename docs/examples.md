# Examples

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
print(f"RMSE: {metrics.rmse(y_test, predictions):.4f}")
```

## Demo Script

Run the demo script to see all features in action:

```bash
python examples/demo.py
```

This will:
1. Generate synthetic data
2. Train all model types
3. Compare their performance
4. Generate visualization plots in `examples/output/`

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
