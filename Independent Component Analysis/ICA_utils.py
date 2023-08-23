import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation


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


def plot_joint_densities(X, S_ica):
    """Plots the joint densities of mixed signals and signals separated by ICA."""

    plt.figure(figsize=(10, 5))

    # Plot mixed signals
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], color='blue')
    plt.title('Joint Density of Mixed Signals')
    plt.xlabel('Mixed Signal 1')
    plt.ylabel('Mixed Signal 2')

    # Plot signals separated by ICA
    plt.subplot(1, 2, 2)
    plt.scatter(S_ica[:, 0], S_ica[:, 1], color='green')
    plt.title('Joint Density After ICA')
    plt.xlabel('ICA Signal 1')
    plt.ylabel('ICA Signal 2')

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
