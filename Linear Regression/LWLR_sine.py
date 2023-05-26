"""
This program implements Locally Weighted Linear Regression (LWLR) on a generated dataset.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(42)

# Generate 100 sample points
train_X = np.sort(np.random.rand(100))
train_Y = np.sin(2 * np.pi * train_X) + 0.3 * np.random.randn(100)

# Design Matrix X as Inputs
X = np.column_stack((np.ones_like(train_X), train_X))
# Target Value Vector
Y = train_Y

# Generate 100 Query Points from 0 to 1
x_query = np.linspace(0, 1, 100)

# Set Parameters
tau = 0.1  # bandwidth


def lwlr(x_query, X, Y, tau):
    """Locally Weighted Linear Regression Function"""
    y_query = np.zeros_like(x_query)

    for i, x_q in enumerate(x_query):
        coord_distance = X[:, 1] - x_q
        w = np.exp(-coord_distance**2 / (2 * tau**2))
        w = np.diag(w)
        # ! theta = (X.T*w*X)^-1*X.T*w*Y
        theta = np.linalg.inv(X.T @ w @ X) @ X.T @ w @ Y
        y_query[i] = x_q * theta[1] + theta[0]

    return y_query


# Output Y From x_query
y_query = lwlr(x_query, X, Y, tau)

# Plot the generated dataset
plt.scatter(train_X, train_Y, c='b', marker='o', label='Target')
plt.plot(x_query, y_query, 'r', label='LWLR Prediction')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Local Weighted Linear Regression')
plt.legend()
plt.show()
