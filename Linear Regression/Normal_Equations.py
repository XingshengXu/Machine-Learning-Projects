import numpy as np
import matplotlib.pyplot as plt


class NormalEquations:
    """
    LWLR is a non-parametric algorithm that fits the model to the data at prediction time, 
    using a weight matrix to give higher importance to data points near the point of interest.

    Args:
        tau (float): The bandwidth parameter which determines the range of influence of the data points.

    Attributes:
        X (np.array): The input training dataset.
        y (np.array): The target values.
    """

    def __init__(self):
        self.theta = np.zeros(2)
        self.IsFitted = False

    def fit(self, X, y):
        """Train the model with the given training set."""

        self.X = np.column_stack((np.ones_like(X), X))
        self.Y = y

        # Theta Found by Normal Functions to minimizes the mean squared error
        # ! theta = (X.T*X)^-1*X.T*Y
        self.theta = np.dot(np.dot(np.linalg.inv(
            np.dot(self.X.T, self.X)), self.X.T), self.Y)
        print(
            f"Theta values: theta_0 = {self.theta[0]}, theta_1 = {self.theta[1]}")
        self.IsFitted = True

    def predict_value(self, X):
        """Compute the predicted value."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            pred_y = self.theta[0] + self.theta[1] * X
            return pred_y

    def create_regression_plot(self, X, y, pred_y):
        """Create a plot to visualize the Normal Equations predictions."""

        plt.scatter(X, y, c='black', marker='o', label='Target')
        plt.plot(X, pred_y, 'r', label='Normal Equations Prediction')
        plt.xlabel('Input Data')
        plt.ylabel('Predict Value')
        plt.title('Linear Regression (Normal Equations)')
        plt.legend()
        plt.show()
