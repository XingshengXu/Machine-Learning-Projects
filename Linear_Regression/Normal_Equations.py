"""
This program implements a linear regression model using Normal Equations to 
predict house prices based on living area (in square feet).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
house_data = pd.read_csv('Linear_Regression/dataset/house_price.csv')

# Settings
data_size = 100


def predict_price(sqft_living, theta_0, theta_1):
    """Compute the predicted price using the real theta values"""
    predicted_price = theta_0 + theta_1 * sqft_living
    return predicted_price


# Training Set
real_training_set_X = house_data['sqft_living'].head(data_size)
real_training_set_Y = house_data['price'].head(data_size)

# Design Matrix X as Inputs
X = np.column_stack((np.ones_like(real_training_set_X), real_training_set_X))

# Target Value Vector
Y = real_training_set_Y

# Theta Found by Normal Functions to minimizes the mean squared error
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),
               Y)  # ! theta = (X.T*X)^-1*X.T*Y
theta_0, theta_1 = theta
print(f"Theta values: theta_0 = {theta_0}, theta_1 = {theta_1}")

# Predict price from real_training_set_X
predicted_price = predict_price(real_training_set_X, theta_0, theta_1)

# Plot the generated dataset
plt.scatter(real_training_set_X, real_training_set_Y,
            c='black', marker='o', label='Target')
plt.plot(real_training_set_X, predicted_price,
         'r', label='Normal Equations Prediction')
plt.xlabel('Living Area (ft²)')
plt.ylabel('House Price ($)')
plt.title('Linear Regression (Normal Equations)')
plt.legend()
plt.show()
