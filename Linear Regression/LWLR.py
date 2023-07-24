import numpy as np
import matplotlib.pyplot as plt


class LocalWeightedLinearRegression:
    """
    LWLR is a non-parametric algorithm that fits the model to the data at prediction time, 
    using a weight matrix to give higher importance to data points near the point of interest.

    Args:
        tau (float): The bandwidth parameter which determines the range of influence of the data points.

    Attributes:
        X (np.array): The input training dataset.
        y (np.array): The target values.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, tau):
        self.tau = tau
        self.IsFitted = False

    def fit(self, X, y):
        """Train the model with the given training set."""

        self.X = np.column_stack((np.ones_like(X), X))
        self.Y = y
        self.IsFitted = True

    def predict_value(self, x_query):
        """Compute the predicted value."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            y_query = np.zeros_like(x_query)

            for i, x_q in enumerate(x_query):
                coord_distance = self.X[:, 1] - x_q
                w = np.exp(-coord_distance**2 / (2 * self.tau**2))
                w = np.diag(w)
                # ! theta = (X.T*w*X)^-1*X.T*w*Y
                theta = np.linalg.inv(
                    self.X.T @ w @ self.X) @ self.X.T @ w @ self.Y
                y_query[i] = x_q * theta[1] + theta[0]

            return y_query

    def create_regression_plot(self, X, y, x_query, y_query):
        """Create a plot to visualize the LWLR predictions."""

        plt.scatter(X, y, c='black', marker='o', label='Target')
        plt.plot(x_query, y_query, 'r', label='LWLR Prediction')
        plt.xlabel('Input Data')
        plt.ylabel('Predict Value')
        plt.title('Local Weighted Linear Regression')
        plt.legend()
        plt.show()
