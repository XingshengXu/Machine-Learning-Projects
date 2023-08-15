import numpy as np


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) is a dimensionality reduction technique that 
    projects data into a lower-dimensional space. It identifies the axes in the 
    dataset that maximize variance and projects the data onto these axes to reduce 
    the dataset's dimensions.

    The number of dimensions to reduce to is determined based on a variance 
    threshold such that a specified percentage (e.g., 95%) of the total variance 
    in the data is retained.

    Args:
        variance_threshold (float): The desired percentage of total variance to be 
                                    retained after dimensionality reduction. 

    Attributes:
        components (numpy.ndarray): Principal components derived from the data.
    """

    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold

    def standardize(mean, X):
        """Standardize the dataset."""

        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        standardize_X = (X - mean) / std_dev
        return standardize_X

    def PCA_transform(self, X):
        """Transform the original dataset to its PCA representation."""

        # Standardize the data
        standardize_X = self.standardize(X)

        # Compute the covariance matrix
        covariance_matrix = standardize_X.T @ standardize_X

        # Use Singular Value Decomposition to calculate the principal components of the data.
        U, sigma, V = np.linalg.svd(covariance_matrix)
        self.components = V

        # Compute the explained variance for each component
        explained_variance = sigma / np.sum(sigma)

        # Compute the cumulative explained variance
        cumulative_variance = np.cumsum(explained_variance)

        # Find number of components needed to retain the desired variance
        component_number = np.argmax(
            cumulative_variance >= self.variance_threshold) + 1

        # Extract the required number of principal components.
        transform_matrix = V.T[:, :component_number]

        # Project the data onto the selected principal components.
        return X @ transform_matrix
