import matplotlib.pyplot as plt
import numpy as np
import path_setup
from K_Nearest_Neighbors import KNN


def plot_lle_embedding(X, y):
    """Plot the 2D data after LLE transformation."""

    # Set up the figure
    fig, ax = plt.subplots()

    # Scatter plot of data points
    ax.scatter(X[:, 0], X[:, 1], c=y,
               cmap='coolwarm', edgecolors='k', s=50)

    ax.set_xlabel('LLE Component 1')
    ax.set_ylabel('LLE Component 2')
    ax.set_title('2D Projection of LLE Dimension Reduced Data')

    plt.show()


def plot_3d_data(X, y):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    if y is None:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=50)
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                             c=y, cmap='coolwarm', s=50, edgecolors='k')
        # Create a legend based on the colors (useful if y represents clusters or classes)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Original Input Data Plot')

    plt.show()


class LocallyLinearEmbedding:
    """
    Locally Linear Embedding (LLE) is a manifold learning technique that reduces the dimensionality 
    of the data by preserving the local linear relationships between data points.

    Args:
        n_neighbors (int): The number of neighbors to consider for each point.
        n_components (int): The number of dimensions to reduce to.

    Attributes:
        n_neighbors (int): The number of neighbors to consider for each point.
        n_components (int): The number of dimensions to reduce to.
    """

    def __init__(self, k_neighbors=5, d_components=2):
        self.k_neighbors = k_neighbors
        self.d_components = d_components

    def fit_transform(self, X):
        # Find the k-nearest neighbors
        knn = KNN(self.k_neighbors)
        knn.fit(X, y=None)

        weight_matrix = np.zeros((X.shape[0], X.shape[0]))

        for i, x in enumerate(X):
            knn.find_k_nearest_neighbours(x)
            neighbors_indices = knn.neighbour_idx

            # Compute the weights
            neighbor_differences = X[neighbors_indices] - x
            covariance = neighbor_differences @ neighbor_differences.T
            covariance_reg = covariance + np.eye(
                self.k_neighbors) * 1e-3 * np.trace(covariance)
            local_weights = np.linalg.solve(
                covariance_reg, np.ones(self.k_neighbors))
            local_weights /= np.sum(local_weights)
            weight_matrix[i, neighbors_indices] = local_weights

        # Compute the embedding
        mat_diff = (np.eye(X.shape[0]) - weight_matrix)
        cost_matrix = mat_diff.T @ mat_diff
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cost_matrix)
        lle_embedding = self.eigenvectors[:, 1:self.d_components+1]
        return lle_embedding
