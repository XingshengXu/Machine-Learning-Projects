"""
Locally Weighted Linear Regression (LWLR) for predicting house prices based on living area.

This program reads a CSV file containing house prices and living area information, and uses
Locally Weighted Linear Regression to predict house prices based on the living area. The
predictions are plotted against the actual house prices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
house_data = pd.read_csv('Linear_Regression/dataset/house_price.csv')

# Settings
data_size = 100

# Training Set
training_set_X = house_data['sqft_living'].head(data_size)
training_set_Y = house_data['price'].head(data_size)

# Design Matrix X as Inputs
X = np.column_stack((np.ones_like(training_set_X), training_set_X))
# Target Value Vector
Y = training_set_Y

# Generate 100 Query Points from 0 to 4000 ft²
x_query = np.linspace(1000, 4000, 100)

# Set Parameters
tau = 300  # bandwidth


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
plt.scatter(training_set_X, training_set_Y, c='b', marker='o', label='Target')
plt.plot(x_query, y_query, 'r', label='LWLR Prediction')
plt.xlabel('Living Area (ft²)')
plt.ylabel('House Price ($)')
plt.title('Local Weighted Linear Regression')
plt.legend()
plt.show()
