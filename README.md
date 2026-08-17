# Simple Linear Regression — From Scratch

A from-scratch implementation of **Ordinary Least Squares (OLS) simple linear regression** in Python — no `sklearn.linear_model`, no shortcuts. Built to understand the math behind the model before relying on library abstractions.

## Overview

This project implements the closed-form OLS solution for fitting a line `y = mx + b` to data, using only the covariance/variance formula:

```
m = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ((xᵢ - x̄)²)
b = ȳ - m·x̄
```

The model is validated on a synthetic dataset — **CGPA vs. Placement Package (LPA)** — where the true underlying relationship is known in advance, allowing direct verification that the fitted parameters recover the true signal despite added noise.

## Why build this from scratch?

Libraries like `scikit-learn` hide the mechanics of linear regression behind a `.fit()` call. Implementing OLS manually forces an understanding of:

- How the slope and intercept are derived from covariance and variance
- Why the least-squares solution minimizes squared error
- What assumptions (linearity, homoscedasticity) the model relies on

This is a first-principles building block before moving to regularized models (Ridge, Lasso) and more complex regressors used later in the portfolio.

## Dataset

The dataset is **synthetically generated**, not hand-typed, to guarantee correctness and reproducibility:

```python
np.random.seed(42)
m_true, b_true = 2.5, -10

CGPA = np.linspace(5, 10, 50)                  # 50 evenly spaced CGPA values
noise = np.random.normal(0, 0.5, 50)            # Gaussian noise, σ = 0.5
Package_LPA = m_true * CGPA + b_true + noise    # true linear relationship + noise
```

- **50 samples**, CGPA range 5.0–10.0
- True relationship: `Package_LPA = 2.5 × CGPA − 10`
- Gaussian noise (σ = 0.5) added to simulate real-world variance — a perfectly clean line would validate the code but not the model's robustness to noise

## Results

| Metric | Value |
|---|---|
| Recovered slope (m) | 2.4173 (true: 2.5) |
| Recovered intercept (b) | -9.4703 (true: -10) |
| R² Score | 0.9785 |
| RMSE | 0.4486 LPA |
| MAE | 0.3611 LPA |

The model recovers parameters close to the true `m` and `b` used to generate the data, confirming the OLS implementation is mathematically correct. The gap between recovered and true values is expected — it reflects the injected noise and finite sample size (50 points), not an error in the formula.

## Project Structure

```
.
├── simple_linear_regression.py   # SimpleLinearRegression class + demo script
└── README.md
```

## Usage

```bash
pip install pandas numpy scikit-learn
python simple_linear_regression.py
```

`scikit-learn` is used only for `train_test_split` — the regression model itself has zero dependency on `sklearn.linear_model`.

## Implementation Notes

- `fit()` computes `m` and `b` via a loop-based covariance/variance calculation (deliberately unvectorized for clarity — a NumPy-vectorized version is a natural next optimization)
- `predict()` applies the fitted line to new inputs
- Train/test split: 80/20, `random_state=42` for reproducibility

## Possible Extensions

- Vectorize `fit()` using NumPy array operations instead of a Python loop
- Add a residual plot to visualize error distribution
- Extend to multiple linear regression (multi-feature `X`)
- Compare against `sklearn.linear_model.LinearRegression` to confirm identical coefficients

---

Built as a foundational ML project — part of an ongoing self-directed AI/ML engineering roadmap.
