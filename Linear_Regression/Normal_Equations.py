import numpy as np
import pandas as pd

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
X = np.vstack((np.ones_like(real_training_set_X), real_training_set_X)).T

# Target Value Vector
Y = real_training_set_Y

# Theta Found by Normal Functions to minimizes the mean squared error
theta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T),
               Y)  # ! theta = (X.T*X)^-1*X.T*Y
theta_0, theta_1 = theta
print(f"Theta values: theta_0 = {theta_0}, theta_1 = {theta_1}")


# Example usage: Predict the price of a house with 2000 sqft of living space
sqft_living = 2000
predicted_price = predict_price(sqft_living, theta_0, theta_1)
print(
    f"The predicted price for a house with {sqft_living} sqft of living space is: ${predicted_price:.2f}")
