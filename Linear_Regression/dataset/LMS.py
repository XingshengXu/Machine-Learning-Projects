import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

# Load Data
house_data = pd.read_csv('Linear_Regression/dataset/house_price.csv')

# Settings
theta_0, theta_1 = 0, 0
data_size = 100
iteration = 0
cost = float('inf')
learning_rate = 0.1
max_iterations = 100
tolerance = 0.001
cost_memo = []


def normalize(data):
    """Normalization Function"""
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev


def h_func(theta_0, theta_1, x):
    """Hypothesis Function"""
    return theta_0 + theta_1 * x


# Training Set

real_training_set_X = house_data['sqft_living'].head(data_size)
real_training_set_Y = house_data['price'].head(data_size)
training_set_X, X_mean, X_dev = normalize(
    house_data['sqft_living'].head(data_size))
training_set_Y, Y_mean, Y_dev = normalize(house_data['price'].head(data_size))

# Initialize the plot
fig, ax = plt.subplots()
ax.scatter(real_training_set_X, real_training_set_Y, c='black', label='Target')
ax.set_xlabel('Living Area (ft\u00B2)')
ax.set_ylabel('House Price ($)')
ax.set_title('Linear Regression')
line, = ax.plot(real_training_set_X, h_func(theta_0, theta_1,
                (real_training_set_X - X_mean) / X_dev) * Y_dev + Y_mean, 'r-', label='Predict Line')
plt.legend()


def update(frame):
    """Update function for the animation"""
    global theta_0, theta_1, cost, iteration
    train_theta_0 = 0
    train_theta_1 = 0
    cost_res = 0
    for i in range(data_size):
        y_pred = h_func(theta_0, theta_1, training_set_X[i])
        y_true = training_set_Y[i]
        error = y_true - y_pred
        train_theta_0 += error
        train_theta_1 += error * training_set_X[i]
        cost_res += error ** 2

    theta_0 += learning_rate * train_theta_0 / data_size
    theta_1 += learning_rate * train_theta_1 / data_size
    cost = cost_res / (2 * data_size)
    cost_memo.append(cost)
    iteration += 1

    line.set_ydata(h_func(theta_0, theta_1,
                   (real_training_set_X - X_mean) / X_dev) * Y_dev + Y_mean)
    return line,


# Animation settings
ani = FuncAnimation(fig, update, frames=max_iterations, interval=50, blit=True)

# Display the animation
plt.show()

# Save the animation as a GIF
# writer = PillowWriter(fps=20)
# ani.save('Linear Regression of House Price.gif', writer=writer)


print(f"Training finished after {iteration} iterations.")
print(f"Theta values: theta_0 = {theta_0}, theta_1 = {theta_1}")
print(cost)

# Plot cost vs. iteration
fig2, ax2 = plt.subplots()
ax2.plot(range(1, iteration + 1), cost_memo)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_title('Cost vs. Iteration')
plt.show()
