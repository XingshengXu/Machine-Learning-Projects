"""Hand Written Digits Image Noise Reduction Using Hopfield Network"""

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def sign_func(x):
    """ Sign Function"""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return x


def flipping_noise(image, flip_fraction):
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


# Load training data sets
try:
    train_set = idx2numpy.convert_from_file(
        'Hopfield Neural Network/dataset/train-images.idx3-ubyte')
    label_set = idx2numpy.convert_from_file(
        'Hopfield Neural Network/dataset/train-labels.idx1-ubyte')
except FileNotFoundError as e:
    print("One or more data files not found.")
    print(e)
    exit()

# Parameters
threshold = 128
pattern_size = 3
pattern_index = 2
is_equal = False
train_index = []

# Pick training images
for digit in range(pattern_size):
    digit_indices = np.where(label_set == digit)[0]
    train_index.append(digit_indices[0])

train_images = train_set[train_index]

# Transpose the matrix so that the image indices are the third dimension
X_train = np.transpose(train_images, (1, 2, 0))

# Reshape the images (28*28) to intput data (784*1)
X_train = np.reshape(X_train, (784, pattern_size))

# Transfer greyscale images to bipolar images
X_train = np.where(X_train > threshold, 1, -1)

# Hopfield neural network training
weights = X_train @ X_train.T

# Keep the diagonal elements to be zero
np.fill_diagonal(weights, 0)

# Hopfield neural network noise reduction
Y_test_prev = flipping_noise(X_train[:, pattern_index], 0.1)

# Initialize previous and current states
Y_test_curr = np.copy(Y_test_prev)

# Vectorize the sign function and apply it to network
sign_func_vec = np.vectorize(sign_func)

# Create a figure for the animation
fig, ax = plt.subplots(figsize=(5, 5))

# Adjust the size of the plot to remove whitespace
plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                    hspace=0, wspace=0)

# Remove axes
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())

# The function to update the figure for each frame of the animation


def update(frame):
    global Y_test_prev, Y_test_curr, is_equal, i_neuron
    ax.clear()

    # Update the next neuron
    i_neuron = frame % len(Y_test_curr)
    Y_test_curr[i_neuron] = sign_func_vec(weights[i_neuron] @ Y_test_prev)
    Y_test_prev = np.copy(Y_test_curr)

    # Plot the current state
    output_image = Y_test_curr.reshape(28, 28)
    img = ax.imshow(output_image, cmap='gray')
    ax.axis('off')
    return img,


# Determine the total number of frames (one per neuron update)
total_frames = len(Y_test_curr)

# Create the animation
ani = FuncAnimation(fig, update, frames=total_frames,
                    interval=1, blit=True, repeat=False)

# Save the animation as a GIF
writer = PillowWriter(fps=60)
ani.save('Hopfield Neural Network/Image Noise Reduction (digit 2).gif', writer=writer)

# Display the animation
plt.show()
