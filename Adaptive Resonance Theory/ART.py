import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.signal import wiener
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def generate_test_data(n_samples=100, n_features=2, std=0.5):
    """Generate a test dataset with n-dimensional instances."""

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2,
                      random_state=0, cluster_std=std)

    # Normalization of input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).T

    # Complement coding of input data to prevent the weight decreases too fast
    X = np.vstack((X, 1.0 - X))
    return X, y


def complement_coding(input_data):
    """This function calculates the complement of a given input data 
    and stacks it horizontally with the original input data."""

    # Ensure the input is a numpy array.
    input_data = np.array(input_data)

    # Calculate the complement of the input matrix.
    complement_data = 1.0 - input_data

    # Stack the original input matrix and its complement horizontally.
    return np.vstack((input_data, complement_data))


def create_contour_plot(art, X, y, resolution=1000, alpha=0.5):
    """Plot the decision boundary of the clusters formed by the ART model"""

    # Generate a grid of points over the actual range of the training data
    x_min, y_min = X[:2, :].min(axis=1) - 0.1
    x_max, y_max = X[:2, :].max(axis=1) + 0.1

    x_values, y_values = np.meshgrid(np.linspace(x_min, x_max, resolution),
                                     np.linspace(y_min, y_max, resolution))

    # Create an empty array to hold the predicted labels of each point on the grid
    pred_labels = np.zeros((resolution, resolution))

    for i in range(resolution):
        for j in range(resolution):
            data_point = np.array([x_values[i, j], y_values[i, j],
                                   1-x_values[i, j], 1-y_values[i, j]])
            pred_labels[i, j] = art.predict_label(
                data_point, art.cluster_labels)

    plt.figure(figsize=(10, 7))
    plt.contourf(x_values, y_values, pred_labels, alpha=alpha, cmap='bwr')

    # Plot the training data, color-coded based on their true label
    plt.scatter(X[0, :], X[1, :], c=y, edgecolors='k', cmap='bwr')

    plt.xlabel('Feature One')
    plt.ylabel('Feature Two')
    plt.title('Fuzzy Adaptive Resonance Theory Classifier')

    plt.show()


def preprocess_image(image, block_shape):
    """Preprocesses the input image for compression."""

    # Convert the image to greyscale
    img_grey = image.convert('L')

    # Convert the image data to a numpy array and Normalize input image
    train_image = np.asarray(img_grey) / 255

    # Reshape the image into 4D space of designated size of blocks
    image_blocks = train_image.reshape(train_image.shape[0] // block_shape[0], block_shape[0],
                                       train_image.shape[1] // block_shape[1], block_shape[1])

    # Combine and reshape each block into designated size of vectors
    reshaped_blocks = image_blocks.transpose(
        0, 2, 1, 3).reshape(-1, block_shape[0] * block_shape[1])

    # Transpose to get the final input matrix
    train_X = reshaped_blocks.T

    # Complement coding
    train_X = complement_coding(train_X)

    # Generate Labels (y) only used to generate ART Class instance
    train_Y = np.ones(train_X.shape[1], dtype='int')

    return train_image, train_X, train_Y


def run_length_encoding(input_string):
    """Performs run-length encoding on the input  code string."""
    count = 1
    prev = ""
    code = []
    for character in input_string:
        if character != prev:
            if prev:
                entry = (prev, count)
                code.append(entry)
            count = 1
            prev = character
        else:
            count += 1
    entry = (prev, count)
    code.append(entry)
    return 2 * len(code)


def decode_compressed_image(art, train_image, block_shape):
    """Decodes the compressed image using Code Book and Block Codes."""

    # Discomplement coding for blocks to form the Block Codes
    trained_blocks = art.weights[:np.prod(block_shape), :]

    # Compute the shape of the grid of blocks
    grid_shape = (train_image.shape[0] // block_shape[0],
                  train_image.shape[1] // block_shape[1])

    # Calculate the length of Code Book after RLE
    length_after_RLE = run_length_encoding(art.cluster_id)

    # Reshape the Code Book into a 2D grid
    cluster_id_grid = art.cluster_id.reshape(grid_shape)

    # Initialize the compressed image
    compressed_image = np.zeros(train_image.shape)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # Find the index for the current block
            cluster_index = cluster_id_grid[i, j]

            # Get the corresponding Block Code
            code_block = trained_blocks[:, cluster_index]

            # Reshape the Block Code to the original shape of a block
            reshaped_block = code_block.reshape(block_shape)

            # Place the decoded block in the correct position in the image
            compressed_image[i * block_shape[0]:(i + 1) * block_shape[0],
                             j * block_shape[1]:(j + 1) * block_shape[1]] = reshaped_block

    # Denormalize the compressed image
    compressed_image = compressed_image * 255

    # Apply Wiener smoothing filter
    compressed_image = wiener(compressed_image)

    return compressed_image, length_after_RLE, trained_blocks


def create_image_plot(art, train_image, compressed_image, block_shape):
    """Creates a plot showing the original image and the compressed image."""
    # Plot original image and compressed image
    plt.figure(figsize=(10, 5))

    # Create the first subplot for the original image
    plt.subplot(1, 2, 1)
    plt.imshow(train_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # Create the second subplot for the compressed image
    plt.subplot(1, 2, 2)
    plt.imshow(compressed_image, cmap='gray')
    plt.title("Compressed Image")
    plt.axis('off')

    suptitle_text = (
        f"Fuzzy Adaptive Resonance Theory Based Image Compression\n"
        f"(block size: {block_shape[0]}x{block_shape[1]} "
        f"learning rate: {art.learning_rate} vigilance parameter: {art.epsilon})"
    )
    plt.suptitle(suptitle_text)
    plt.show()


def evaluate_compression(train_image, compressed_image, length_after_RLE, trained_blocks):
    """Evaluates the compression performance by calculating compression ratio, 
    MSE, PSNR, and plotting the compression difference heatmap."""

    # Calculate compression ratio
    compression_ratio = np.prod(train_image.shape) / (
        length_after_RLE + np.prod(trained_blocks.shape))

    # Calculate MSE
    mse = mean_squared_error(train_image, compressed_image)

    # Calculate PSNR
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # Calculate difference
    difference = np.abs(train_image - compressed_image)

    # Plot heatmap of compression difference
    plt.imshow(difference, cmap='binary', interpolation='nearest')
    plt.axis('off')

    # Add a colorbar to the right side
    colorbar = plt.colorbar(orientation='vertical', pad=0.02)
    colorbar.set_label('Difference')

    title_text = f'Heatmap of compression difference\n' \
        f'(Compression Ratio: {compression_ratio:.2f}, ' \
        f'MSE: {mse:.2f}, ' \
        f'PSNR: {psnr:.2f})'
    plt.title(title_text)
    plt.show()


class AdaptiveResonanceTheory():
    """Implementation of the Fuzzy Adaptive Resonance Theory (ART) clustering algorithm.
    ART is a form of unsupervised learning. It is designed to quickly categorize input patterns 
    and maintain the stability of recognized clusters in response to changing inputs. 
    The term "adaptive resonance" refers to the system's capacity to self-adjust (adapt) 
    and create a "resonance", or state of harmony, when a new or familiar input signal is presented.

    Args:
        learning_rate (float, optional): The learning rate of the algorithm which determines how fast 
                                         the weights are updated. Default is 0.1.
        alpha (int, optional): The choice parameter, contributes to the denominator in the choice function 
                               to prevent very small or very large input vectors from dominating the learning 
                               process. Default is 1.
        epsilon (float, optional): The vigilance parameter, determines how strict the model is when assigning 
                                   input vectors to existing clusters. Default is 0.6.

    Attributes:
        iteration (int): Current iteration count.
        IsEqual (bool): Boolean flag to check whether cluster assignments have changed.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
        weights (numpy.ndarray): Weight matrix where each column represents the weight vector for a cluster.
        cluster_id (numpy.ndarray): Array that holds the cluster id for each instance in the training set.
        valid_indices (list): List of indices for clusters that have at least one associated instance.
        num_valid_clusters (list): A list to hold the number of valid clusters during the model fitting.
    """

    def __init__(self, learning_rate=0.1, alpha=1, epsilon=0.6):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.iteration = 0
        self.IsEqual = False
        self.IsFitted = False
        self.cluster_changes = []
        self.num_valid_clusters = []

    def choice_function(self, X_train):
        """Computes the choice function which ranks clusters in order of fitness to be selected."""
        #! T_j = Σ min(X_i, w_ji) / (α + Σ w_ji)
        min_matrix = np.minimum(self.weights, X_train)
        choice_score = min_matrix.sum(axis=0) / (self.alpha + self.weight_sum)
        return np.argsort(choice_score)[::-1]

    def vigilance_test(self, X_train, candidate_index):
        """Computes the vigilance test which verifies if the selected cluster matches 
        closely enough with the input vector."""
        #! S_j = Σ min(X_i, w_ji) / Σ X_i

        min_matrix = np.minimum(self.weights[:, candidate_index], X_train)
        match_score = np.sum(min_matrix) / np.sum(X_train)

        if match_score >= self.epsilon:  # If match score meets the vigilance criterion

            #! Update weights of the neuron: w_ji_new = β * min(w_ji_old, X_i) + (1-β) * w_ji_old
            self.weights[:, candidate_index] = self.learning_rate * min_matrix + (
                1 - self.learning_rate) * self.weights[:, candidate_index]
            self.weight_sum = self.weights.sum(axis=0)
            return True
        return False

    def count_valid_clusters(self):
        """Count the number of unique elements in cluster_id. This gives the number 
        of clusters that have at least one instance associated with them"""
        unique_clusters = np.unique(self.cluster_id)
        valid_cluster_count = len(unique_clusters)
        return valid_cluster_count

    def fit(self, X, y):
        """Fits the model using input vectors and corresponding labels.
        Note, the input date shall have the form of (n_features, n_samples)."""

        self.cluster_id = np.zeros(X.shape[1], dtype=np.int32)
        self.weights = X[:, 0].reshape(-1, 1)
        self.weight_sum = self.weights.sum(axis=0)

        while not self.IsEqual:
            cluster_id_prev = self.cluster_id.copy()

            for i in range(X.shape[1]):
                # Compute choice function
                candidate_indices = self.choice_function(X[:, [i]])

                # Vigilance test
                for candidate_index in candidate_indices:
                    if self.vigilance_test(X[:, i], candidate_index):
                        self.cluster_id[i] = candidate_index
                        break
                else:
                    self.weights = np.column_stack((self.weights, X[:, i]))
                    self.cluster_id[i] = self.weights.shape[1] - 1
                    self.weight_sum = self.weights.sum(axis=0)

            self.IsEqual = np.array_equal(cluster_id_prev, self.cluster_id)
            self.iteration += 1
            self.num_valid_clusters.append(self.count_valid_clusters())
            changes = np.count_nonzero(self.cluster_id != cluster_id_prev)
            self.cluster_changes.append(changes)
            print(
                f'Iteration {self.iteration}: {np.count_nonzero(self.cluster_id != cluster_id_prev)} different clusters')

        # Compute labels for each cluster
        self.cluster_labels = [np.argmax(np.bincount(y[self.cluster_id == id_value])) if np.any(
            self.cluster_id == id_value) else None for id_value in range(self.weights.shape[1])]

        # The model has been fitted
        self.IsFitted = True

    def predict(self, X):
        '''Return the models predicted cluster for each of the given instances.'''

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            predicted_labels = [self.predict_label(
                test_data, self.cluster_labels) for test_data in X.T]
        return predicted_labels

    def predict_label(self, X_test, cluster_labels):
        """Predicts the label of an instance by finding the cluster with the highest choice function value."""

        X_test = X_test.reshape(-1, 1)
        min_matrix = np.minimum(self.weights, X_test)
        choice_score = min_matrix.sum(axis=0) / (self.alpha + self.weight_sum)

        # Only consider clusters that have a label
        self.valid_indices = [i for i in range(
            len(cluster_labels)) if cluster_labels[i] is not None]
        valid_scores = choice_score[self.valid_indices]

        # Find the cluster with the highest choice function value
        best_match_index = self.valid_indices[np.argmax(valid_scores)]

        # Assign the label of the best match cluster to the test example
        return cluster_labels[best_match_index]

    def predict_test(self, test_Y, pred_Y):
        """This function prints the classification report and plots the confusion 
        matrix for the given actual and predicted labels."""

        # Print classification report
        print(classification_report(test_Y, pred_Y, zero_division=0))

        # Create confusion matrix
        cm = confusion_matrix(test_Y, pred_Y)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))

        # Plot the number of changes among clusters after each iteration
        axs[0].plot(range(self.iteration), self.cluster_changes,
                    color='red', alpha=0.7, linewidth=2.5)
        axs[0].set(xlabel='Iteration', ylabel='Changes Among Clusters')
        axs[0].set_title('Changes Among Clusters over Iterations')

        # Plot the number of valid clusters after each iteration
        axs[1].plot(range(self.iteration), self.num_valid_clusters,
                    color='blue', alpha=0.7, linewidth=2.5)
        axs[1].set(xlabel='Iteration', ylabel='Number of Valid Clusters')
        axs[1].set_title('Number of Valid Clusters over Iterations')

        # Adjust spacing between subplots
        fig.tight_layout()
        plt.show()
