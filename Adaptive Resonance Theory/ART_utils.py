import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from scipy.signal import wiener
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from Adaptive_Resonance_Theory import AdaptiveResonanceTheory


def generate_test_data(n_samples=100, n_features=2, centers=2, std=0.5):
    """Generate a test dataset with n-dimensional instances."""

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers,
                      random_state=0, cluster_std=std)

    # Normalization of input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Complement coding of input data to prevent the weight decreases too fast
    X = np.hstack((X, 1.0 - X))
    return X, y


def complement_coding(input_data):
    """This function calculates the complement of a given input data 
    and stacks it horizontally with the original input data."""

    # Ensure the input is a numpy array.
    input_data = np.array(input_data)

    # Calculate the complement of the input matrix.
    complement_data = 1.0 - input_data

    # Stack the original input matrix and its complement horizontally.
    return np.hstack((input_data, complement_data))


def create_contour_plot(art, X, y, resolution=1000, alpha=0.5):
    """Plot the decision boundary of the clusters formed by the ART model"""

    # We only use the first two dimensions of X for plotting.
    X_plot = X[:, :2]

    # Generate a grid of points over the actual range of the training data
    x_min, y_min = X_plot.min(axis=0) - 0.1
    x_max, y_max = X_plot.max(axis=0) + 0.1

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
    plt.contourf(x_values, y_values, pred_labels, alpha=alpha, cmap='jet')

    # Plot the training data, color-coded based on their true label
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, edgecolors='k', cmap='jet')

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

    # Complement coding
    train_X = complement_coding(reshaped_blocks)

    # Generate Labels (y) only used to generate ART Class instance
    train_Y = np.ones(train_X.shape[0], dtype='int')

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
    trained_blocks = art.weights[:, :np.prod(block_shape)]

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
            code_block = trained_blocks[cluster_index, :]

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


def interactive_data_collection_classification():
    """
    Create an interactive plot for collecting data points for a Classification Task.

    The function allows you to interactively add points of two classes by left-clicking
    for class 0 and right-clicking for class 1. It also allows you to train an Adaptive 
    Resonance Theory Classifier and visualize the decision boundaries by clicking the 
    'Train' button, or to clear all data points and start over by clicking the 'Clean' 
    button.
    """

    # Initialize click coordinates and labels
    coords, labels = [], []

    # Set color for each class
    class_colors = ['blue', 'red']

    # Create an interactive plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set plot properties
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("$\mathbf{Adaptive\ Resonance\ Theory\ Classifier\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
                 "\nLeft click to input 'class 1' data, and right click to input 'class 2' data.")

    # Define onclick function for collecting data
    def onclick(event):
        if event.inaxes == ax:
            label = 0 if event.button == 1 else 1
            coords.append((event.xdata, event.ydata))
            labels.append(label)
            ax.scatter(event.xdata, event.ydata,
                       c=class_colors[label], edgecolors='black')
            fig.canvas.draw()

    # Define onpress function for training model
    def onpress(event):
        if 0 in labels and 1 in labels:
            X, y = np.array(coords), np.array(labels)

            # Complement coding of input data to prevent the weight decreases too fast
            X = np.hstack((X, 1.0 - X))
            art = AdaptiveResonanceTheory(
                learning_rate=0.1,
                alpha=0.5,
                epsilon=0.9
            )
            art.fit(X, y)
            create_contour_plot(art, X, y, resolution=500)

    # Define onclean function for resetting data
    def onclean(event):
        coords.clear()
        labels.clear()

        # Remove drawn elements
        for coll in (ax.collections + ax.lines):
            coll.remove()
        fig.canvas.draw()

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("$\mathbf{Adaptive\ Resonance\ Theory\ Classifier\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
                     "\nLeft click to input 'class 1' data, and right click to input 'class 2' data.")
        fig.canvas.draw()

    # Create 'Train' and 'Clean' buttons
    ax_button_train = plt.axes([0.25, 0.01, 0.2, 0.06])
    button_train = Button(ax_button_train, 'Train')
    button_train.on_clicked(onpress)

    ax_button_clear = plt.axes([0.55, 0.01, 0.2, 0.06])
    button_clear = Button(ax_button_clear, 'Clean')
    button_clear.on_clicked(onclean)

    # Register onclick function
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
