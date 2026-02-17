# Mathematics

## Ordinary Least Squares (Normal Equation)

```
θ = (XᵀX)⁻¹Xᵀy
```

## Gradient Descent

```
repeat until convergence:
    θ := θ - α * ∇J(θ)
```

## Ridge Regression

```
θ = (XᵀX + λI)⁻¹Xᵀy
```

## Lasso Regression

```
minimize: (1/2n) * ||y - Xw||² + α * ||w||₁
```

## ElasticNet

```
minimize: (1/2n) * ||y - Xw||² + α * (l1_ratio * ||w||₁ + 0.5 * (1 - l1_ratio) * ||w||²)
```
