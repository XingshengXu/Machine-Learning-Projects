import matplotlib.pyplot as plt
import numpy as np
import path_setup
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from K_Nearest_Neighbors import KNN


def plot_mvu_embedding(X, y, t=None):
    """Plot the 2D data after MVU transformation."""

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 7))

    if y is None and t is not None:
        ax.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.rainbow)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap='coolwarm', edgecolors='k', s=50)

    ax.set_xlabel('MVU Component 1')
    ax.set_ylabel('MVU Component 2')
    ax.set_title('2D Projection of MVU Dimension Reduced Data')

    plt.show()


def plot_3d_data(X, y=None, t=None, rotate=False):
    """Plot the orignal 3D input data."""

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if y is None and t is not None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.rainbow)
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                             c=y, cmap='coolwarm', s=50, edgecolors='k')
        # Create a legend based on the colors
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Original Input Data Plot')

    if rotate:
        # Rotate 90 degrees along the y-axis
        ax.view_init(elev=0, azim=-90)

    plt.show()


class MaximumVarianceUnfolding:
    """
    Maximum Variance Unfolding (MVU) is a dimensionality reduction technique 
    akin to Principal Component Analysis (PCA). While PCA focuses on finding 
    the principal components that maximize variance, MVU focuses on maximizing 
    the variance of pairwise distances between data points, often in 
    semi-supervised or unsupervised contexts.

    MVU seeks to unfold the data manifold such that the variance in pairwise 
    distances is maximized, typically in lower-dimensional spaces.

    Args:
        k_neighbors (int, optional): The number of neighbors to consider.
        d_components (int, optional): Target dimensionality.
    Attributes:
        embedding_ (numpy.ndarray): Lower-dimensional embedding of the data.
    """

    def __init__(self, k_neighbors=30, d_components=2):
        self.k_neighbors = k_neighbors
        self.d_components = d_components

    def fit_transform(self, X):
        """Fit MVU algorithm and return transformed data."""

        # Find the k-nearest neighbors
        knn = KNN(self.k_neighbors)
        knn.fit(X, y=None)

        n_samples = X.shape[0]
        pairwise_distances = np.zeros((n_samples, n_samples))

        for i, x in enumerate(X):
            knn.find_k_nearest_neighbours(x)
            pairwise_distances[i, knn.neighbour_idx] = np.linalg.norm(
                X[knn.neighbour_idx] - x, axis=1)

        # Ensure symmetry in pairwise_distances
        pairwise_distances = (pairwise_distances + pairwise_distances.T) / 2

        # Convert pairwise distances into Gram matrix
        J = np.eye(n_samples) - (1 / n_samples) * \
            np.ones((n_samples, n_samples))
        H = -0.5 * J @ (pairwise_distances ** 2) @ J

        # Ensure symmetry in H
        H = 0.5 * (H + H.T)

        # Regularize
        reg_value = 1e-6
        H += np.eye(n_samples) * reg_value

        # Eigendecomposition
        eigenvalues, eigenvectors = eigsh(
            csr_matrix(H), k=self.d_components+1, which='LM')

        # Extract embedding
        mvu_embedding = eigenvectors[:, :self.d_components] @ np.diag(
            np.sqrt(np.abs(eigenvalues[:self.d_components])))

        return mvu_embedding
