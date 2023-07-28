import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


class LinearRegression:
    """
    Linear Regression using either Stochastic Gradient Descent (SGD) or Batch Gradient Descent (BGD)
    where a linear model is trained to predict output values by iteratively adjusting 
    its parameters based on the individual sample or the entire dataset to minimize the difference 
    between predicted and actual output values.

    Args:
        algorithm (str): 'BGD' for Batch Gradient Descent, 'SGD' for Stochastic Gradient Descent
        learning_rate (float, optional): Learning rate for Gradient Descent.
        max_iterations (int, optional): Maximum number of iterations for the Gradient Descent loop.

    Attributes:
        theta (np.array): Model weights.
        theta_real (np.array): Real-scale model weights.
        iteration (int): Counter for iterations.
        cost (float): Current cost value.
        cost_memo (list): Memory of cost at each iteration.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, algorithm='BGD', learning_rate=0.1, max_iterations=100):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.theta = np.zeros(2)
        self.theta_real = np.zeros(2)
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def normalize(self, data):
        """Normalization Function."""

        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        normalized_data = (data - mean) / std_dev
        return normalized_data, mean, std_dev

    def h_func(self, X):
        """Hypothesis Function."""

        return np.dot(X, self.theta)

    def h_func_real(self, X):
        """Hypothesis Function for real-scale data."""

        return self.theta_real[0] + self.theta_real[1] * X

    def predict_value(self, X):
        """Compute the predicted value using the real theta values."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            pred_y = self.h_func_real(X)
            return pred_y

    def initialize_plot(self, X, y):
        """Initialize plot for animation."""

        self.fig, self.ax = plt.subplots()
        self.ax.scatter(X, y, c='black', label='Target')
        self.ax.set_xlabel('Input X')
        self.ax.set_ylabel('Output y')
        self.line, = self.ax.plot(
            X, self.h_func_real(X), 'r-', label='Predict Line')
        plt.legend()

    def update_BGD(self, frame):
        """Update function for BGD."""

        pred_y = self.h_func(self.normalized_X)
        error = self.normalized_y - pred_y
        grad = np.dot(self.normalized_X.T, error)
        self.theta += self.learning_rate * grad / self.X.shape[0]

        # Update theta values based on the real-scale data
        self.theta_real[1] = (self.y_dev / self.X_dev) * self.theta[1]
        self.theta_real[0] = self.y_mean - self.theta_real[1] * self.X_mean

        self.cost = np.sum(error ** 2) / (2 * self.X.shape[0])
        self.cost_memo.append(self.cost)
        self.iteration += 1

        self.line.set_ydata(self.h_func_real(self.X))
        return self.line,

    def update_SGD(self, frame):
        """Update function for SGD."""

        # Shuffle the data at each iteration
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)

        self.cost = 0

        for i in indices:
            X_sample = self.normalized_X[i]
            y_sample = self.normalized_y[i]

            y_pred = self.h_func(X_sample)
            error = y_sample - y_pred
            grad = X_sample * error
            self.theta += self.learning_rate * grad

            # Update theta values based on the real-scale data
            self.theta_real[1] = (self.y_dev / self.X_dev) * self.theta[1]
            self.theta_real[0] = self.y_mean - self.theta_real[1] * self.X_mean

            # Calculate and accumulate the cost for each sample
            self.cost += error ** 2 / 2

        # Append the total cost of the iteration to the memo
        self.cost_memo.append(self.cost / self.X.shape[0])

        self.iteration += 1

        self.line.set_ydata(self.h_func_real(self.X))
        return self.line,

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (n_samples, n_features).
        """

        self.X = X
        self.normalized_X, self.X_mean, self.X_dev = self.normalize(X)
        self.normalized_y, self.y_mean, self.y_dev = self.normalize(y)

        # Add a column of ones to the training set for theta_0
        self.normalized_X = np.vstack(
            (np.ones_like(self.normalized_X), self.normalized_X)).T

        # Initialize plot for animation
        self.initialize_plot(X, y)

        if self.algorithm == 'BGD':
            self.update = self.update_BGD
            self.ax.set_title('Linear Regression (Batch Gradient Descent)')
        elif self.algorithm == 'SGD':
            self.update = self.update_SGD
            self.ax.set_title(
                'Linear Regression (Stochastic Gradient Descent)')
        else:
            raise ValueError('Invalid algorithm type. Choose "BGD" or "SGD".')

        # Animation settings
        self.animation = FuncAnimation(self.fig, self.update, frames=self.max_iterations,
                                       interval=50, blit=True, repeat=False)

        # *Save animation as gif
        # writer = PillowWriter(fps=20)
        # self.animation.save(
        #     f'./Linear Regression ({self.algorithm}) of Synthetic Data Set.gif', writer=writer)

        # Display the animation
        plt.show()

        # The model has been fitted
        self.IsFitted = True
