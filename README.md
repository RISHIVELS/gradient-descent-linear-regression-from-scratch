# Gradient Descent Linear Regression (from scratch)

Lightweight, dependency-free linear regression implemented with batch gradient descent. The project demonstrates fitting height–weight data using only NumPy and a few plotting libraries, with clear visibility into parameter and loss trajectories.

## Project structure

- notebooks/gradient_descent_exploration.ipynb — exploratory analysis and model walkthrough
- src/linear_regression_gd.py — core implementation of `LinearRegressionGradientDescent`
- data/height_weight.csv — sample dataset (height in inches, weight in pounds)
- requirements.txt — Python dependencies

## Model overview

- Hypothesis: $\hat y = \beta_0 + \beta_1 x$
- Loss (per epoch): Mean Squared Error $J = \frac{1}{2n} \sum_{i=1}^n (\hat y_i - y_i)^2$
- Gradients: $\frac{\partial J}{\partial \beta_0} = \frac{1}{n}\sum (\hat y - y)$, $\frac{\partial J}{\partial \beta_1} = \frac{1}{n}\sum (\hat y - y)x$
- Update rule: $\beta_j \leftarrow \beta_j - \alpha \cdot \frac{\partial J}{\partial \beta_j}$
- Convergence: stop early when parameter deltas are below `tol`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt
```

## Quickstart (script or notebook)

```python
import pandas as pd
import numpy as np
from src.linear_regression_gd import LinearRegressionGradientDescent

# Load data
df = pd.read_csv('data/height_weight.csv')
df = df.rename(columns={'Height(Inches)': 'height', 'Weight(Pounds)': 'weight'})
x = df['height'].to_numpy()
y = df['weight'].to_numpy()

# Train
model = LinearRegressionGradientDescent()
model.fit(x, y, alpha=1e-4, epochs=20000)

# Evaluate
y_pred = model.predict(x)
model.summary(y, y_pred)

# Parameters
print(model.beta0, model.beta1)
```

## Notes on hyperparameters

- Learning rate (`alpha`): the default `1e-4` is safe for the unscaled height feature (~70 range). If you normalize features, you can increase `alpha` (e.g., `0.01`).
- Epochs: increase (e.g., 50,000) if convergence is slow; the loop stops early once deltas fall below `tol`.
- Tolerance (`tol`): smaller values yield more precise convergence at the cost of runtime.

## Inspecting training dynamics

The model records parameter and loss history on each epoch:

- `beta0_history`, `beta1_history`: intercept and slope trajectory
- `slope_history`: gradient for `beta1` each step
- `loss_history`: MSE/2 per epoch

These arrays can be plotted (see notebook Cell 23–25) to visualize convergence and the loss surface.

## Metrics

- RMSE, MAE, MSE, R² are provided in `summary(y, y_pred)` and individually via helper methods.

## Troubleshooting

- Diverging loss or NaNs: lower `alpha` or standardize the feature.
- Slow convergence: increase `epochs` or scale features to zero mean/unit variance.
- Shape issues: ensure `x` and `y` are 1-D arrays of the same length before calling `fit`.
