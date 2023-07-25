import numpy as np
from sklearn.preprocessing import StandardScaler


class GaussianDiscriminant:
    """
    Gaussian Discriminant Analysis (GDA) is a generative learning algorithm used for 
    binary classification. It assumes that the input data for each class is generated 
    from a Gaussian distribution, and it estimates the parameters of these distributions 
    to make predictions.

    Attributes:
        X (np.array): Standardized feature matrix (training data).
        y (np.array): Output vector (training labels).
        mean_y0 (np.array): Mean vector for class 0.
        mean_y1 (np.array): Mean vector for class 1.
        covariance_matrix (np.array): Shared covariance matrix.
        covariance_matrix_inv (np.array): Inverse of the shared covariance matrix.
        covariance_matrix_det (float): Determinant of the shared covariance matrix.
        const_factor (float): Constant factor for the multivariate Gaussian distribution.
        prob_y0 (float): Prior probability for class 0.
        prob_y1 (float): Prior probability for class 1.
        p_x_given_y0 (np.array): Probability densities for class 0.
        p_x_given_y1 (np.array): Probability densities for class 1.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self):
        self.IsFitted = False

    def fit(self, X, y):
        """Train the model with the given training set."""

        # Standardize the features of input data
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X).T
        self.y = y

        # Get number of features and training samples
        data_dim, train_data_size = self.X.shape

        # Swap 0s with 1s and 1s with 0s in output set for summation of indicator functions
        swapped_y = np.where(self.y == 0, 1, 0)

        # The parameters of the GDA model
        # ! φ = 1/n * ΣI(y(i)=1)
        prior_prob = 1 / train_data_size * np.sum(self.y)

        # ! μ0 = ΣI(y(i)=0) * x(i) / ΣI(y(i)=0)
        mean_y0 = self.X @ swapped_y / np.sum(swapped_y)

        # ! μ0 = ΣI(y(i)=1) * x(i) / ΣI(y(i)=1)
        mean_y1 = self.X @ self.y / np.sum(self.y)

        # Reshape means to be column vectors
        self.mean_y0 = mean_y0.reshape(-1, 1)
        self.mean_y1 = mean_y1.reshape(-1, 1)

        # Subtract means from input X based on corresponding y
        X_mu = self.X.copy()

        # ! Σ = 1/n * Σ((x(i) - μy(i)) * (x(i) - μy(i))^T)
        X_mu[:, self.y == 0] -= self.mean_y0
        X_mu[:, self.y == 1] -= self.mean_y1

        # Compute the covariance matrix
        self.covariance_matrix = (X_mu @ X_mu.T) / train_data_size

        # Distributions for two classes
        self.prob_y0 = 1 - prior_prob
        self.prob_y1 = prior_prob

        # Compute the inverse and determinant of the covariance matrix
        self.covariance_matrix_inv = np.linalg.inv(self.covariance_matrix)
        self.covariance_matrix_det = np.linalg.det(self.covariance_matrix)

        # Constant factor for the multivariate guassian distribution
        self.const_factor = 1 / ((2 * np.pi) ** (data_dim / 2)
                                 * np.sqrt(self.covariance_matrix_det))

        self.IsFitted = True

    def predict_class(self, X):
        """Predict the class for given input data."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            # Standardize the features of input data
            X = self.scaler.transform(X).T

            # Subtract the corresponding means from the input data
            diff_y0 = X - self.mean_y0
            diff_y1 = X - self.mean_y1

            # Compute exponential part
            exp_arg0 = np.sum((diff_y0.T @ self.covariance_matrix_inv)
                              * diff_y0.T, axis=1)
            exp_arg1 = np.sum((diff_y1.T @ self.covariance_matrix_inv)
                              * diff_y1.T, axis=1)

            # Compute the probability densities

            # ! p(x|y=0) = (1 / (2π)^(n/2) * |Σ|^(1/2)) * exp(-1/2 * (x - μ0)^T * Σ^-1 * (x - μ0))
            self.p_x_given_y0 = self.const_factor * np.exp(-0.5 * exp_arg0)

            # ! p(x|y=1) = (1 / (2π)^(n/2) * |Σ|^(1/2)) * exp(-1/2 * (x - μ1)^T * Σ^-1 * (x - μ1))
            self.p_x_given_y1 = self.const_factor * np.exp(-0.5 * exp_arg1)

            # Predict the classes based on the probability densities
            pred_y = self.prob_y1 * self.p_x_given_y1 > self.prob_y0 * self.p_x_given_y0
            return pred_y
