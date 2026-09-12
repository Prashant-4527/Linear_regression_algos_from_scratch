# Linear Regression From Scratch

This folder contains first-principles implementations of linear regression in Python. The goal is to understand how a regression model learns its coefficients before using a library such as `sklearn.linear_model.LinearRegression`.

The project covers:

- Simple linear regression with one feature
- Multiple linear regression with several features
- Gradient descent as an alternative optimization method

## What is linear regression?

Linear regression predicts a continuous target by learning a weighted sum of its input features.

For one feature:

```text
y_hat = m x + b
```

For multiple features:

```text
y_hat = b + w1 x1 + w2 x2 + ... + wn xn
```

The model chooses the weights that minimize the mean squared error (MSE):

```text
MSE = (1 / n) sum((y - y_hat)^2)
```

This is called the least-squares objective because large prediction errors receive a larger penalty after squaring.

## Algorithms implemented

### 1. Simple linear regression

`simple_linear_regression.py` fits a line using the closed-form ordinary least squares equations:

```text
m = sum((x - x_mean)(y - y_mean)) / sum((x - x_mean)^2)
b = y_mean - m x_mean
```

The script creates a reproducible synthetic dataset that relates CGPA to placement package in LPA, splits it into training and test sets, fits the model, and prints predictions.

### 2. Multiple linear regression

`multi_linear_regression.py` fits a model with more than one input feature using the normal equation:

```text
beta = (X^T X)^-1 X^T y
```

The implementation adds a column of ones to `X` so that the first learned coefficient is the intercept. The remaining coefficients are stored in `coef_`.

### 3. Gradient descent

`gdregressor.py` contains the starting structure for a gradient-descent regressor. Gradient descent repeatedly updates the parameters in the direction that reduces the loss instead of solving the normal equation directly.

This file is currently a work in progress and is not yet runnable. It is kept as the next implementation step for studying iterative optimization.

## Project structure

```text
.
├── gdregressor.py                 # Gradient descent draft
├── multi_linear_regression.py     # Multiple linear regression
├── simple_linear_regression.py    # Single-feature OLS demo
└── README.md
```

## Requirements

- Python 3.9 or newer
- NumPy
- pandas
- scikit-learn, used only by the demo for `train_test_split`

Install the dependencies with:

```bash
pip install numpy pandas scikit-learn
```

## Run the simple regression demo

```bash
python simple_linear_regression.py
```

The regression class itself does not use `sklearn.linear_model`. The only scikit-learn feature in the demo is the train/test split.

## Example workflow

The custom classes follow the familiar `fit()` and `predict()` pattern:

```python
import numpy as np
from multi_linear_regression import MultiLinearRegression

X = np.array([
	[1.0, 2.0],
	[2.0, 1.0],
	[3.0, 4.0],
])
y = np.array([5.0, 6.0, 11.0])

model = MultiLinearRegression()
model.fit(X, y)
predictions = model.predict(X)
```

## Important assumptions and limitations

Linear regression works best when:

- The relationship between inputs and the target is approximately linear.
- Observations are independent.
- The input features are not perfectly collinear.
- The variance of the errors is reasonably consistent.

The normal-equation implementation uses a matrix inverse. That can fail when `X^T X` is singular or be numerically unstable for poorly conditioned data. A production implementation would generally use `np.linalg.solve()` or a pseudoinverse instead.

The current classes also expect NumPy arrays with compatible shapes and do not yet include validation, scaling, metrics, or plotting utilities.

## Planned improvements

- Complete and test the gradient-descent implementation.
- Add input-shape and fitted-model validation.
- Replace explicit matrix inversion with a numerically safer solver.
- Add MSE, RMSE, MAE, and R-squared metrics.
- Add tests for exact data, noisy data, multiple features, and edge cases.
- Compare the learned coefficients with scikit-learn as a verification step.

This project is intended as a learning implementation: the equations are kept visible so the connection between the mathematics and the code remains clear.
