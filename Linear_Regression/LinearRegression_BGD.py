"""
Batch Gradient Descent (BGD) for predicting house prices based on living area.

This code implements a linear regression model using Batch Gradient Descent (BGD) 
to predict the price of a house based on its living area (sqft_living). The model is trained 
on a house price dataset loaded from a CSV file, and the real-scale theta values are computed 
after normalizing the data. The training process is animated using `matplotlib`'s FuncAnimation 
class, and the cost vs. iteration curve is plotted after training. The prediction for the price 
of a house with 2000 sqft of living space is also computed as an example.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

# Load Data
house_data = pd.read_csv('Linear_Regression/dataset/house_price.csv')

# Settings
theta = np.zeros(2)
theta_0_real, theta_1_real = np.zeros(2)
data_size = 100
iteration = 0
cost = np.inf
learning_rate = 0.1
max_iterations = 100
cost_memo = []


def normalize(data):
    """Normalization Function"""
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev


def h_func(theta, x):
    """Hypothesis Function"""
    return theta[0] + theta[1] * x


def h_func_real(theta_0, theta_1, x):
    """Hypothesis Function for real-scale data"""
    return theta_0 + theta_1 * x


def predict_price(sqft_living, theta_0_real, theta_1_real):
    """Compute the predicted price using the real theta values"""
    predicted_price = h_func_real(theta_0_real, theta_1_real, sqft_living)
    return predicted_price


# Training Set
real_training_set_X = house_data['sqft_living'].head(data_size)
real_training_set_Y = house_data['price'].head(data_size)
training_set_X, X_mean, X_dev = normalize(
    house_data['sqft_living'].head(data_size))
training_set_Y, Y_mean, Y_dev = normalize(house_data['price'].head(data_size))

# Add a column of ones to the training set for theta_0
X = np.vstack((np.ones_like(training_set_X), training_set_X)).T

# Initialize the plot
fig, ax = plt.subplots()
# Use real_training_set_X and real_training_set_Y
ax.scatter(real_training_set_X, real_training_set_Y, c='black', label='Target')
ax.set_xlabel('Living Area (ft²)')
ax.set_ylabel('House Price ($)')
ax.set_title('Linear Regression (Batch Gradient Descent)')
line, = ax.plot(real_training_set_X, h_func_real(theta_0_real, theta_1_real, real_training_set_X),
                'r-', label='Predict Line')  # Use h_func_real and real_training_set_X
plt.legend()


def update(frame):
    """Update function for the animation"""
    global theta, cost, iteration, theta_0_real, theta_1_real
    y_pred = h_func(theta, training_set_X)
    error = training_set_Y - y_pred
    grad = np.dot(X.T, error)
    theta += learning_rate * grad / data_size

    # Update theta values based on the real-scale data
    theta_1_real = (Y_dev / X_dev) * theta[1]
    theta_0_real = Y_mean - theta_1_real * X_mean

    cost = np.sum(error ** 2) / (2 * data_size)
    cost_memo.append(cost)
    iteration += 1
    # Use h_func_real and real_training_set_X
    line.set_ydata(h_func_real(
        theta_0_real, theta_1_real, real_training_set_X))
    return line,


# Animation settings
ani = FuncAnimation(fig, update, frames=max_iterations,
                    interval=50, blit=True, repeat=False)

# Save the animation as a GIF
# writer = PillowWriter(fps=20)
# ani.save('Linear_Regression/Linear Regression of House Price.gif', writer=writer)

# Display the animation
plt.show()

print(f"Training finished after {iteration} iterations.")
print(f"Theta values: theta_0 = {theta_0_real}, theta_1 = {theta_1_real}")

# Plot cost vs. iteration
fig2, ax2 = plt.subplots()
ax2.plot(range(1, iteration + 1), cost_memo)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_title('Cost vs. Iteration')
plt.show()

# Example usage: Predict the price of a house with 2000 sqft of living space
sqft_living = 2000
predicted_price = predict_price(sqft_living, theta_0_real, theta_1_real)
print(
    f"The predicted price for a house with {sqft_living} sqft of living space is: ${predicted_price:.2f}")
