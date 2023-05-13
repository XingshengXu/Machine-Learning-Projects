"""
Logistic Regression for classificating marketing target.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

# Load Data
market_data = pd.read_csv(
    'Logistic_Regression/dataset/Social_Network_Ads.csv', header=0)

# Settings
theta = np.zeros(4)
data_size = 400
iteration = 0
cost = np.inf
learning_rate = 0.001
tolerence = 0.5
cost_memo = []


def h_func(theta, X):
    """Hypothesis Function"""
    return 1 / (1 + np.exp(-theta.T @ X))


def cost_func(theta, X, Y):
    """Cost Function"""
    h = h_func(theta, X)
    return -1 * np.mean(Y * np.log(h) + (1 - Y) * np.log(1 - h))


# Training Set
training_set_X = market_data.iloc[:, 1:4]
training_set_Y = market_data.iloc[:, 4]

# Change Unit in 'EstimatedSalary' Column as $k
training_set_X['EstimatedSalary'] /= 1000

# Change 'Gender' Column as 'Male':1 and 'Female':0
training_set_X['Gender'] = training_set_X['Gender'].replace(
    {'Male': 1, 'Female': 0})

# Add A Column of Ones to The Training Set As Intercept Term
X = np.hstack((np.ones((data_size, 1)), training_set_X)).T

# Impletement Logistic Regression Training

# Loop through the entire dataset for each epoch
while cost >= tolerence:
    for i in range(data_size):
        X_sample = X[:, i]
        y_sample = training_set_Y[i]
        y_pred = h_func(theta, X_sample)
        error = y_sample - y_pred
        grad = X_sample * error
        theta += learning_rate * grad
    cost = cost_func(theta, X, training_set_Y)
    cost_memo.append(cost)
    iteration += 1

print(f"Training finished after {iteration} iterations.")
print(f"Theta values:{theta}")

# Plot cost vs. iteration
plt.plot(range(1, iteration + 1), cost_memo, 'o')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()
