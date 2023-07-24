import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def preprocess_image(image, label, pattern_size):
    """Preprocesses the input image for training."""

    train_index = []
    threshold = 128

    for digit in range(pattern_size):
        digit_indices = np.where(label == digit)[0]
        train_index.append(digit_indices[0])

    train_images = image[train_index]

    # Transpose the matrix so that the image indices are the third dimension
    X = np.transpose(train_images, (1, 2, 0))

    # Reshape the images (28*28) to intput data (784*1)
    X = np.reshape(X, (784, pattern_size))

    # Transfer grayscale images to bipolar images
    X = np.where(X > threshold, 1, -1)
    return X


class HopfieldNetwork:
    """
    An implementation of a Hopfield network, a type of artificial neural network that 
    serves as a form of content-addressable memory system. Using a simple learning rule, 
    the Hopfield network can learn binary patterns and later retrieve them when provided 
    with a noisy or partial version of those patterns, making it especially useful for 
    tasks such as noise reduction in images.

    Attributes:
        iteration (list): Records the iteration at which each pattern becomes stable.
        image_memo (list): Stores the states of the image at each iteration.
        is_equal (bool): Boolean flag to indicate if the current and previous states are equal.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self):
        self.iteration = []
        self.image_memo = []
        self.is_equal = False
        self.IsFitted = False

    def sign_func(self, x):
        """Build sign function."""

        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return x

    def flipping_noise(self, image, flip_fraction):
        """
        This function takes an image and a fraction representing the amount of noise
        to add and returns a noisy image. The noise is simply flipping of pixels.
        """

        # Flatten the image if it is not
        if len(image.shape) > 1:
            image = image.flatten()

        # Generate a mask for flipping pixels
        noise_mask = np.random.choice([False, True], len(
            image), p=[1-flip_fraction, flip_fraction])

        # Return a copy of the image with the pixels at the mask indices flipped
        return np.where(noise_mask, -image, image)

    def fit(self, X, y):
        """Train the model with the given training set."""

        pattern_size = X.shape[1]

        # Hopfield neural network training
        self.weights = X @ X.T

        # Keep the diagonal elements to be zero
        np.fill_diagonal(self.weights, 0)

        for pattern_index in range(pattern_size):
            self.is_equal = False
            iteration_pattern = 0
            image_memo_pattern = []

            # Hopfield neural network noise reduction
            prev_y = self.flipping_noise(X[:, pattern_index], 0.1)

            # Vectorize the sign function and apply it to network
            sign_func_vec = np.vectorize(self.sign_func)

            while not self.is_equal:
                curr_y = sign_func_vec(self.weights @ prev_y)

                # Check if the output is at its stable state
                self.is_equal = np.array_equal(prev_y, curr_y)
                image_memo_pattern.append(prev_y)
                prev_y = curr_y
                iteration_pattern += 1

            self.image_memo.append(image_memo_pattern)
            self.iteration.append(iteration_pattern)

        self.IsFitted = True

    def plot_image(self):
        """Plot the noise reduction process for each image pattern."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            for pattern_index in range(len(self.iteration)):
                plt.figure(figsize=(10, 5))

                for i in range(self.iteration[pattern_index]):
                    plt.subplot(1, self.iteration[pattern_index], i+1)
                    output_image = self.image_memo[pattern_index][i].reshape(
                        28, 28)
                    plt.imshow(output_image, cmap='gray')
                    plt.suptitle(
                        f'Hand Written Digits Image Noise Reduction Using Hopfield Network: Pattern {pattern_index}')
                    plt.axis('off')

                plt.show()

    def animate_hopfield(self, X, pattern_index):
        """Creates an animation showing the noise reduction process for a specific pattern."""

        # Initialize previous and current pattern
        prev_y = self.flipping_noise(X[:, pattern_index], 0.1)
        curr_y = np.copy(prev_y)

        # Vectorize the sign function and apply it to network
        sign_func_vec = np.vectorize(self.sign_func)

        # Create a figure for the animation
        fig, ax = plt.subplots(figsize=(5, 5))

        # Adjust the size of the plot to remove whitespace
        plt.subplots_adjust(top=1, bottom=0, right=1,
                            left=0, hspace=0, wspace=0)

        # Remove axes
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # The function to update the figure for each frame of the animation
        def update(frame):
            nonlocal prev_y, curr_y
            ax.clear()

            # Update the next neuron
            i_neuron = frame % len(curr_y)
            curr_y[i_neuron] = sign_func_vec(
                self.weights[i_neuron] @ prev_y)
            prev_y = np.copy(curr_y)

            # Plot the current state
            output_image = curr_y.reshape(28, 28)
            img = ax.imshow(output_image, cmap='gray')
            ax.axis('off')
            return img,

        # Determine the total number of frames (one per neuron update)
        total_frames = len(curr_y)

        # Create the animation
        ani = FuncAnimation(fig, update, frames=total_frames,
                            interval=1, blit=True, repeat=False)

        # Save the animation as a GIF
        writer = PillowWriter(fps=60)
        ani.save(
            f'Image Noise Reduction (digit {pattern_index}).gif', writer=writer)
        print("Animation has been saved, check your project directory.")
