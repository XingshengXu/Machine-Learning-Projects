import numpy as np


class IndependentComponentAnalysis:
    """
    Independent Component Analysis (ICA) is a blind source separation method 
    which separates a multivariate signal into additive, independent signals.

    Args:
        n_components (int): The number of independent components to extract from the data.

    Attributes:
        components (numpy.ndarray): Independent components derived from the data.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def standardize(self, X):
        """Standardize the dataset."""

        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        standardize_X = (X - mean) / std_dev
        return standardize_X

    def whiten(self, X):
        """Whiten the dataset."""

        cov = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        X_whitened = np.dot(U @ np.diag(1.0 / np.sqrt(S)), U.T @ X.T).T
        return X_whitened

    def fit_transform(self, X):
        """Transform the original dataset to its ICA representation."""

        # Standardize the data
        standardize_X = self.standardize(X)

        # Whiten the data
        whitened_X = self.whiten(standardize_X)

        # Random initial weights
        n, m = whitened_X.shape
        if self.n_components is None:
            self.n_components = m

        W = np.random.rand(self.n_components, m)

        # Iteratively update the weights (a basic example, can use FastICA here)
        for _ in range(1000):  # For simplicity, using fixed number of iterations
            W = np.dot(W, np.dot(whitened_X.T, whitened_X) -
                       np.eye(self.n_components))
            W = np.linalg.qr(W)[0]  # Orthogonalize the weights

        self.components = W

        unmixed_signals = np.dot(whitened_X, W.T)
        return unmixed_signals


class FastICA:
    """
    Fast Independent Component Analysis (FastICA) aims to find independent 
    components in the data by maximizing non-Gaussianity.

    Args:
        n_components (int): The number of independent components to extract from the data.

    Attributes:
        components (numpy.ndarray): Independent components derived from the data.
        mixing (numpy.ndarray): Estimated mixing matrix.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def standardize(self, X):
        """Standardize the dataset."""

        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        standardize_X = (X - mean) / std_dev
        return standardize_X

    def whiten(self, X):
        """Whiten the dataset."""

        cov = np.cov(X, rowvar=False)
        U, S, V = np.linalg.svd(cov)
        X_whitened = np.dot(U @ np.diag(1.0 / np.sqrt(S)), U.T @ X.T).T
        return X_whitened

    def _g(self, x):
        """Compute the function g(x) = tanh(x)."""

        return np.tanh(x)

    def _g_prime(self, x):
        """Compute the derivative g'(x) of g(x) = tanh(x)."""

        return 1 - np.tanh(x)**2

    def fit_transform(self, X):
        """Apply FastICA to dataset X."""

        # Standardize the data
        standardize_X = self.standardize(X)

        # Whiten the data
        whitened_X = self.whiten(standardize_X)

        # Random initial weights
        n, m = whitened_X.shape
        if self.n_components is None:
            self.n_components = m

        W = np.random.rand(self.n_components, m)

        # Iteratively update the weights using FastICA
        for _ in range(1000):  # Using a fixed number of iterations for simplicity
            wx = np.dot(W, whitened_X.T)
            gwx = self._g(wx)
            g_prime_wx = self._g_prime(wx)
            W_new = np.dot(gwx, whitened_X) / n - \
                np.dot(np.diag(g_prime_wx.mean(axis=1)), W)
            W = np.linalg.qr(W_new.T)[0].T
            # Convergence check can be added here

        self.components = W
        self.mixing = np.linalg.pinv(W)

        ica_projection = np.dot(whitened_X, W.T)
        return ica_projection
