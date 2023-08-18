import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def create_pca_animation(X, y, pca):
    """Animate the data and principal components in 3D."""

    # Set up the figure and axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm')

    # Number of retained components based on the variance threshold
    num_components = pca.component_number

    # Plot the retained principal components.
    for index in range(num_components):
        component = pca.components[index]
        comp = np.array([component * 15])
        zeros = np.zeros((1, len(component)))
        comp = np.concatenate((comp, zeros), axis=0)
        ax.plot(comp[:, 0], comp[:, 1], comp[:, 2], label=f'PC: {index + 1}')
    ax.legend()

    # Define update function for animation
    def update(degree):
        # Update the elevation and azimuth of the viewpoint (camera)
        ax.view_init(elev=20, azim=degree)
        return ax

    # Create an animation
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, 360, 15), interval=300, blit=False)

    # Save the animation
    ani.save('PCA_3Ddemo.gif', writer='pillow')
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
        # Create a legend based on the colors (useful if y represents clusters or classes)
        legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
        ax.add_artist(legend1)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title('3D Original Input Data Plot')

    if rotate:
        # Rotate 90 degrees along the x-axis
        ax.view_init(elev=90, azim=0)

    plt.show()


def plot_transformed_data(X, y, t=None):
    """Plot the 2D data after PCA transformation."""

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 7))

    if y is None and t is not None:
        ax.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.rainbow)
    else:
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap='coolwarm', edgecolors='k', s=50)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('2D Projection of PCA Dimension Reduced Data')

    plt.show()


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
        variance_threshold (float): The desired percentage of total variance to be 
                                    retained after dimensionality reduction.
        components (numpy.ndarray): Principal components derived from the data.
        explained_variance (numpy.ndarray): Explained variance for each principal component.
        component_number (int): The number of components needed to retain the desired variance.
    """

    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold

    def standardize(mean, X):
        """Standardize the dataset."""

        mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        standardize_X = (X - mean) / std_dev
        return standardize_X

    def fit_transform(self, X):
        """Transform the original dataset to its PCA representation."""

        # Standardize the data
        standardize_X = self.standardize(X)

        # Compute the covariance matrix
        covariance_matrix = standardize_X.T @ standardize_X

        # Use Singular Value Decomposition to calculate the principal components of the data.
        U, sigma, V = np.linalg.svd(covariance_matrix)

        self.components = V

        # Compute the explained variance for each component
        self.explained_variance = sigma / np.sum(sigma)

        # Compute the cumulative explained variance
        cumulative_variance = np.cumsum(self.explained_variance)

        # Find number of components needed to retain the desired variance
        self.component_number = np.argmax(
            cumulative_variance >= self.variance_threshold) + 1

        # Extract the required number of principal components.
        transform_matrix = V.T[:, :self.component_number]

        # Project the data onto the selected principal components.
        pca_projection = X @ transform_matrix
        return pca_projection
