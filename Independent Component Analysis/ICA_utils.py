import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.colors import LogNorm
from skimage import color


def plot_ICA_signals(S, X, S_ica):
    """Visualize original, mixed, and ICA separated signals."""

    plt.figure(figsize=(7, 10))

    # Original Signals
    plt.subplot(6, 1, 1)
    plt.title("Original Signal 1")
    plt.plot(S[:, 0], color='red', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(6, 1, 2)
    plt.title("Original Signal 2")
    plt.plot(S[:, 1], color='red', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Mixed Signals
    plt.subplot(6, 1, 3)
    plt.title("Mixed Signal 1")
    plt.plot(X[:, 0], color='blue', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(6, 1, 4)
    plt.title("Mixed Signal 2")
    plt.plot(X[:, 1], color='blue', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # ICA Separated Signals
    plt.subplot(6, 1, 5)
    plt.title("Recovered Signal 1 using ICA")
    plt.plot(S_ica[:, 0], color='green', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(6, 1, 6)
    plt.title("Recovered Signal 2 using ICA")
    plt.plot(S_ica[:, 1], color='green', alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Super title for the entire plot
    plt.suptitle("Signal Visualization: Original, Mixed, and ICA Recovered")

    plt.tight_layout()
    plt.show()


def create_ica_animation(time, X, S_ica, save_path="ica_animation.gif", fps=5):
    """Create an animation transitioning between mixed signals and ICA separated signals."""

    fig, ax = plt.subplots()

    # Initial mixed signals plot with new colors
    line1, = ax.plot(time, X[:, 0], color='r')
    line2, = ax.plot(time, X[:, 1], color='b')

    ax.set_title('Independent Component Analysis Demo')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

    def update(num, line1, line2, S_ica, X):
        alpha = num / 30.0
        line1.set_ydata((1 - alpha) * X[:, 0] + alpha * S_ica[:, 0])
        line2.set_ydata((1 - alpha) * X[:, 1] + alpha * S_ica[:, 1])
        return line1, line2

    ani = FuncAnimation(fig, update, frames=30, fargs=[
                        line1, line2, S_ica, X], repeat=False)

    # Save the animation as a GIF
    writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)


def plot_joint_density(X, S_ica):
    """Plot the joint distributions of the original and ICA-separated signals."""

    # Initialize the plot
    plt.figure(figsize=(10, 5))

    # Plot Joint Density Before ICA
    plt.subplot(1, 2, 1)
    x, y = X[:, 0], X[:, 1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=[50, 50])
    hist[hist <= 0] = 1e-5
    hist /= np.sum(hist)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    plt.contourf(xcenters, ycenters, hist, levels=10,
                 cmap='inferno', norm=LogNorm(), vmin=1e-5, vmax=1)
    plt.colorbar(label='Frequency (Log Scale)')
    plt.title('Joint Density Before ICA')
    plt.xlabel('Original Signal 1')
    plt.ylabel('Original Signal 2')

    # Plot Joint Density After ICA
    plt.subplot(1, 2, 2)
    x, y = S_ica[:, 0], S_ica[:, 1]
    hist, xedges, yedges = np.histogram2d(x, y, bins=[50, 50])
    hist[hist <= 0] = 1e-5
    hist /= np.sum(hist)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    plt.contourf(xcenters, ycenters, hist, levels=10,
                 cmap='inferno', norm=LogNorm(), vmin=1e-5, vmax=1)
    plt.colorbar(label='Frequency (Log Scale)')
    plt.title('Joint Density After ICA')
    plt.xlabel('ICA Signal 1')
    plt.ylabel('ICA Signal 2')

    # Show the plot
    plt.tight_layout()
    plt.show()


def extract_patches(image, patch_size=(16, 16), max_patches=5000):
    """Extracts random patches from a given 2D image."""

    image = np.array(image)

    # Remove the alpha channel if it exists
    if image.shape[2] == 4:
        image = image[:, :, :3]

    # Convert the color image to grayscale
    image_gray = color.rgb2gray(image)

    # Get the dimensions of the image
    i_h, i_w = image_gray.shape
    # Get the dimensions of the patches to be extracted
    p_h, p_w = patch_size

    # Initialize an empty list to store the patches
    patches = []

    # Loop to extract 'max_patches' number of patches
    for _ in range(max_patches):
        # Randomly select the top-left corner of the patch
        top_left_x = np.random.randint(0, i_w - p_w)
        top_left_y = np.random.randint(0, i_h - p_h)

        # Extract the patch
        patch = image_gray[top_left_y:top_left_y +
                           p_h, top_left_x:top_left_x + p_w]

        # Append the patch to the list
        patches.append(patch)

    # Convert the list of patches to a 3D numpy array
    patches = np.array(patches)

    # Reshape patches to a 2D array suitable for ICA
    n_patches, p_h, p_w = patches.shape

    # Plotting the original grayscale image
    plt.imshow(image_gray, cmap='gray')
    plt.title('Origial Grayscale Image')
    plt.axis('off')
    plt.show()

    return patches.reshape(n_patches, p_h * p_w)
