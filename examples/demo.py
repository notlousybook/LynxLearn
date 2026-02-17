"""
Demo script showing all features of LynxLearn.

This script demonstrates:
- LinearRegression (OLS)
- GradientDescentRegressor
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Polynomial Regression
- Model selection (train_test_split)
- Metrics (MSE, RMSE, MAE, R²)
- Visualizations (basic and comprehensive)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lynxlearn.linear_model import (
    LinearRegression,
    GradientDescentRegressor,
    Ridge,
    Lasso,
    ElasticNet,
    PolynomialRegression,
    PolynomialFeatures,
)
from lynxlearn.model_selection import train_test_split
from lynxlearn import metrics
from lynxlearn import visualizations

np.random.seed(42)

print("=" * 60)
print("LynxLearn - Linear Regression Demo")
print("=" * 60)

# Generate synthetic data
print("\n1. Generating synthetic data...")
n_samples = 200
X = np.random.randn(n_samples, 1) * 5  # Single feature
true_weights = 2.5
true_bias = 5.0
noise = np.random.randn(n_samples) * 2
y = X.flatten() * true_weights + true_bias + noise

print(f"   Data shape: X={X.shape}, y={y.shape}")
print(f"   True weights: {true_weights}, True bias: {true_bias}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Train size: {len(y_train)}, Test size: {len(y_test)}")

# ============================================
# 1. Linear Regression (OLS)
# ============================================
print("\n" + "=" * 60)
print("2. Linear Regression (OLS - Normal Equation)")
print("=" * 60)

lr = LinearRegression()
lr.train(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"   Weights: {lr.weights}")
print(f"   Bias: {lr.bias:.6f}")
print(f"   MSE: {metrics.mse(y_test, y_pred_lr):.6f}")
print(f"   RMSE: {metrics.rmse(y_test, y_pred_lr):.6f}")
print(f"   MAE: {metrics.mae(y_test, y_pred_lr):.6f}")
print(f"   R² Score: {metrics.r2_score(y_test, y_pred_lr):.6f}")
print(f"   Model Score: {lr.score(X_test, y_test):.6f}")

# ============================================
# 2. Gradient Descent Regressor
# ============================================
print("\n" + "=" * 60)
print("3. Gradient Descent Regressor")
print("=" * 60)

gd = GradientDescentRegressor(learning_rate=0.01, n_iterations=1000, tolerance=1e-6)
gd.train(X_train, y_train)
y_pred_gd = gd.predict(X_test)

print(f"   Weights: {gd.weights}")
print(f"   Bias: {gd.bias:.6f}")
print(f"   Iterations: {gd.n_iter_}")
print(f"   Final Cost: {gd.cost_history[-1]:.6f}")
print(f"   MSE: {metrics.mse(y_test, y_pred_gd):.6f}")
print(f"   RMSE: {metrics.rmse(y_test, y_pred_gd):.6f}")
print(f"   MAE: {metrics.mae(y_test, y_pred_gd):.6f}")
print(f"   R² Score: {metrics.r2_score(y_test, y_pred_gd):.6f}")

# ============================================
# 3. Ridge Regression
# ============================================
print("\n" + "=" * 60)
print("4. Ridge Regression (L2 Regularization)")
print("=" * 60)

ridge = Ridge(alpha=1.0)
ridge.train(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"   Weights: {ridge.weights}")
print(f"   Bias: {ridge.bias:.6f}")
print(f"   Alpha (regularization): {ridge.alpha}")
print(f"   MSE: {metrics.mse(y_test, y_pred_ridge):.6f}")
print(f"   RMSE: {metrics.rmse(y_test, y_pred_ridge):.6f}")
print(f"   MAE: {metrics.mae(y_test, y_pred_ridge):.6f}")
print(f"   R² Score: {metrics.r2_score(y_test, y_pred_ridge):.6f}")

# ============================================
# 4. Lasso Regression
# ============================================
print("\n" + "=" * 60)
print("5. Lasso Regression (L1 Regularization)")
print("=" * 60)

lasso = Lasso(alpha=0.1)
lasso.train(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print(f"   Weights: {lasso.weights}")
print(f"   Bias: {lasso.bias:.6f}")
print(f"   Alpha (regularization): {lasso.alpha}")
print(f"   Non-zero weights: {np.sum(lasso.weights != 0)}")
print(f"   MSE: {metrics.mse(y_test, y_pred_lasso):.6f}")
print(f"   RMSE: {metrics.rmse(y_test, y_pred_lasso):.6f}")
print(f"   MAE: {metrics.mae(y_test, y_pred_lasso):.6f}")
print(f"   R² Score: {metrics.r2_score(y_test, y_pred_lasso):.6f}")

# ============================================
# 5. ElasticNet Regression
# ============================================
print("\n" + "=" * 60)
print("6. ElasticNet Regression (L1 + L2 Regularization)")
print("=" * 60)

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.train(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)

print(f"   Weights: {elastic.weights}")
print(f"   Bias: {elastic.bias:.6f}")
print(f"   Alpha: {elastic.alpha}, L1 Ratio: {elastic.l1_ratio}")
print(f"   Non-zero weights: {np.sum(elastic.weights != 0)}")
print(f"   MSE: {metrics.mse(y_test, y_pred_elastic):.6f}")
print(f"   RMSE: {metrics.rmse(y_test, y_pred_elastic):.6f}")
print(f"   MAE: {metrics.mae(y_test, y_pred_elastic):.6f}")
print(f"   R² Score: {metrics.r2_score(y_test, y_pred_elastic):.6f}")

# ============================================
# 6. Model Comparison
# ============================================
print("\n" + "=" * 60)
print("7. Model Comparison")
print("=" * 60)

models = {
    "LinearRegression": lr,
    "GradientDescent": gd,
    "Ridge": ridge,
    "Lasso": lasso,
    "ElasticNet": elastic,
}

print(f"{'Model':<20} {'MSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
print("-" * 68)
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = metrics.mse(y_test, y_pred)
    rmse = metrics.rmse(y_test, y_pred)
    mae = metrics.mae(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    print(f"{name:<20} {mse:<12.6f} {rmse:<12.6f} {mae:<12.6f} {r2:<12.6f}")

# ============================================
# 7. Visualizations
# ============================================
print("\n" + "=" * 60)
print("8. Generating Visualizations")
print("=" * 60)

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Regression plot
    print("   Creating regression plot...")
    fig = visualizations.plot_regression(
        X_test,
        y_test,
        lr,
        title="Linear Regression - OLS",
        save_path=os.path.join(output_dir, "regression_ols.png"),
    )
    plt.close(fig)

    # 2. Cost history for gradient descent
    print("   Creating cost history plot...")
    fig = visualizations.plot_cost_history(
        gd,
        title="Gradient Descent Convergence",
        save_path=os.path.join(output_dir, "cost_history.png"),
    )
    plt.close(fig)

    # 3. Residuals plot
    print("   Creating residuals plot...")
    fig = visualizations.plot_residuals(
        y_test,
        y_pred_lr,
        title="Residuals - Linear Regression",
        save_path=os.path.join(output_dir, "residuals.png"),
    )
    plt.close(fig)

    # 4. Coefficients comparison
    print("   Creating coefficients comparison...")
    fig = visualizations.plot_coefficients(
        models,
        feature_names=["Feature_1"],
        title="Model Coefficients Comparison",
        save_path=os.path.join(output_dir, "coefficients.png"),
    )
    plt.close(fig)

    # 5. Model comparison
    print("   Creating model comparison plot...")
    fig = visualizations.compare_models(
        X_test,
        y_test,
        models,
        title="Model Predictions Comparison",
        save_path=os.path.join(output_dir, "model_comparison.png"),
    )
    plt.close(fig)

    print(f"   [OK] All basic visualizations saved to: {output_dir}")

except ImportError:
    print("   [!] matplotlib not installed, skipping visualizations")
    print("   Install with: pip install matplotlib")

# ============================================
# 8. Comprehensive Visualizations
# ============================================
print("\n" + "=" * 60)
print("9. Comprehensive Visualizations")
print("=" * 60)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("   Creating comprehensive 1D regression visualization...")
    fig = visualizations.visualize_1d_regression(
        X_test,
        y_test,
        lr,
        title="Comprehensive OLS Analysis",
        save_path=os.path.join(output_dir, "comprehensive_ols.png"),
    )
    plt.close(fig)

    print("   Creating gradient descent analysis...")
    fig = visualizations.visualize_gradient_descent(
        X_train,
        y_train,
        gd,
        title="Gradient Descent Deep Dive",
        save_path=os.path.join(output_dir, "gd_analysis.png"),
    )
    plt.close(fig)

    print("   Creating metrics visualization...")
    fig = visualizations.visualize_all_metrics(
        y_test,
        y_pred_lr,
        title="Complete Metrics for OLS",
        save_path=os.path.join(output_dir, "all_metrics.png"),
    )
    plt.close(fig)

    print("   Creating comprehensive report...")
    report = visualizations.create_comprehensive_report(
        X_train, y_train, X_test, y_test, models, save_dir=output_dir
    )
    print(f"   [OK] Report generated with {len(report['results'])} models")

    print(f"   [OK] All comprehensive visualizations saved to: {output_dir}")

except ImportError as e:
    print(f"   [!] Could not create comprehensive visualizations: {e}")

print("\n" + "=" * 60)
print("Demo Complete!")
print("=" * 60)
print(f"\nCheck the output directory for generated visualizations:")
print(f"   {output_dir}")
