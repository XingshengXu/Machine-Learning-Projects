import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap


class LogisticRegression:
    """
    Logistic Regression using either Batch Gradient Descent (BGD) or Newton-Raphson Method (NM).
    This algorithm is used for binary classification, where a logistic model is trained to predict class labels
    by iteratively adjusting its parameters based on the entire dataset to minimize the cost function.
    This cost function measures the difference between the predicted class probabilities and the actual class labels.

    Args:
        algorithm (str): 'BGD' for Batch Gradient Descent, 'NM' for Newton-Raphson Method
        learning_rate (float, optional): Learning rate for Gradient Descent.
        max_iterations (int, optional): Maximum number of iterations for the optimization loop.

    Attributes:
        theta (np.array): Model weights.
        iteration (int): Counter for iterations.
        cost (float): Current cost value.
        cost_memo (list): Memory of cost at each iteration.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, algorithm='BGD', learning_rate=0.01, max_iterations=1000):
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = 0.001
        self.theta = np.zeros(3)
        self.iteration = 0
        self.cost = 0
        self.cost_prev = self.cost_diff = np.inf
        self.cost_memo = []
        self.IsFitted = False

    def h_func(self, X):
        """Build hypothesis function."""

        return 1 / (1 + np.exp(-self.theta.T @ X))

    def gradient(self, X, y):
        """Gradient of the cost function."""

        h = self.h_func(X)
        return X @ (y - h)

    def cost_func(self, X, y):
        """Build cost function."""

        h = self.h_func(X)
        return -1 * np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def hessian(self, X):
        """Build Hessian matrix."""

        h = self.h_func(X)
        D = np.diag(h * (1 - h))
        return X @ D @ X.T

    def fit(self, X, y):
        """Train the model with the given training set."""

        self.X = X
        self.y = y

        # Standardize the features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # Add a column of ones to training set as intercept term
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X)).T

        # Implement logistic regression training
        while self.cost_diff >= self.tolerance and self.iteration <= self.max_iterations:
            grad = self.gradient(self.X, self.y)

            if self.algorithm == 'NM':
                H = self.hessian(self.X)
                self.theta += np.linalg.inv(H) @ grad  # !Newton-Raphson update
            elif self.algorithm == 'BGD':
                self.theta += self.learning_rate * grad  # !Batch Gradient Descent update
            else:
                raise ValueError(
                    'Invalid algorithm type. Choose "BGD" or "NM".')

            self.cost = self.cost_func(self.X, self.y)
            self.cost_diff = np.abs(self.cost_prev - self.cost)
            self.cost_memo.append(self.cost)
            self.cost_prev = self.cost
            self.iteration += 1

        print(f"Training finished after {self.iteration} iterations.")
        print(f"Theta values:{self.theta}")

        self.IsFitted = True

    def predict_class(self, X):
        """Convert probabilities into classes."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            # Standardize the features
            X = self.scaler.transform(X)

            # Add a column of ones to training set as intercept term
            X = np.hstack((np.ones((X.shape[0], 1)), X)).T

            # Predict the class using hypothesis function
            pred_y = self.h_func(X).round()

            return pred_y

    def create_contour_plot(self, X, y):
        """Create a contour plot for the decision boundary of Logistic Regression."""

        # Standardize the features
        X = self.scaler.transform(X)

        # Add a column of ones to training set as intercept term
        X = np.hstack((np.ones((X.shape[0], 1)), X)).T

        # Create Grid
        x_values = np.linspace(-3, 3, 500)
        y_values = (-self.theta[0] - (self.theta[1]
                    * x_values)) / self.theta[2]

        # Create a Colormap
        cmap = ListedColormap(['red', 'blue'])

        # Plotting the Data
        plt.scatter(X[1, :], X[2, :], c=y, cmap=cmap, edgecolors='k')
        plt.plot(x_values, y_values, label='Decision Boundary', c='green')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Training Data with Decision Boundary')
        plt.legend()
        plt.show()
