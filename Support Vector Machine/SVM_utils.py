import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation
from matplotlib.widgets import Button
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from Support_Vector_Machine import SupportVectorMachine


def generate_test_data(n_samples=100, n_features=2, std=0.5):
    '''Generate a test dataset with n-dimensional instances.'''

    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2,
                      random_state=0, cluster_std=std)
    y = convert_labels(y)
    return X, y


def convert_labels(Y):
    """Convert labels from {0,1} to {-1,1} and change the data type to float."""

    Y[Y == 0] = -1
    return Y.astype(np.float64)


def predict_test(test_Y, pred_Y):
    """This function prints the classification report and plots the confusion 
    matrix for the given actual and predicted labels."""

    # Print Classification Report
    print(classification_report(test_Y, pred_Y, zero_division=0))

    # Create Confusion Matrix
    cm = confusion_matrix(test_Y, pred_Y)

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def create_contour_plot(svm, X, y, kernel='linear', filled=False):
    '''Generate a contour plot for the given model using the dataset X with class labels y.'''

    x_min, x_max = np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5
    y_min, y_max = np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    X_grid = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.decision_function(X_grid).reshape(X1.shape)

    plt.figure(figsize=(10, 7))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if filled:
        cp = plt.contourf(X1, X2, Z, cmap='coolwarm')
        colorbar = plt.colorbar(cp)
        colorbar.set_label('Prediction Confidence')
    else:
        cp = plt.contour(
            X1, X2, Z, [-1, 0, 1], colors=['black', 'fuchsia', 'black'], linewidths=2)
        plt.clabel(cp, inline=True, fontsize=10)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                facecolors='none', edgecolors='green', marker='o', s=100)
    plt.xlabel('Feature One')
    plt.ylabel('Feature Two')
    plt.title(
        f'Support Vector Machine Classifier\nKernel: {kernel} C= {svm.C}')
    plt.show()


def create_3d_animation(svm, X, y):
    """Animate the data and decision boundary by SVM in 3D."""

    # Set up the figure and axis
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of data points
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='coolwarm')

    # Get the range for each feature
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # Create a grid of points within the range of each feature
    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    # Calculate Z values (decision boundary) based on SVM model
    Z = (-svm.b - svm.w[0]*X1 - svm.w[1]*X2) / svm.w[2]

    # Plot the decision boundary surface
    ax.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.5)

    # Define update function for animation
    def update(degree):
        # Update the elevation and azimuth of the viewpoint (camera)
        ax.view_init(elev=20, azim=degree)
        return ax

    # Create an animation
    ani = animation.FuncAnimation(
        fig, update, frames=np.arange(0, 360, 15), interval=300, blit=False)

    # Save the animation
    ani.save('SVM_3Ddemo.gif', writer='pillow')
    plt.show()


def interactive_data_collection_classification():
    """
    Create an interactive plot for collecting data points for a Classification Task.

    The function allows you to interactively add points of two classes by left-clicking
    for class 0 and right-clicking for class 1. It also allows you to train a Support 
    Vector Machine Classifier and visualize the decision boundaries by clicking the 
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
    ax.set_title("$\mathbf{Support\ Vector\ Machine\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
                 "\nLeft click to input 'class 1' data, and right click to input 'class 2' data.")

    # Define onclick function for collecting data
    def onclick(event):
        if event.inaxes == ax:
            if event.button == 1:
                label = -1
                idx = 0
            else:
                label = 1
                idx = 1
            coords.append((event.xdata, event.ydata))
            labels.append(label)
            ax.scatter(event.xdata, event.ydata,
                       c=class_colors[idx], edgecolors='black')
            fig.canvas.draw()

    # Define onpress function for training model
    def onpress(event):
        if -1 in labels and 1 in labels:
            X, y = np.array(coords), np.array(labels)
            y = y.astype(np.float64)
            svm = SupportVectorMachine(kernel='rbf', gamma=0.4, C=1)
            svm.fit(X, y)
            create_contour_plot(svm, X, y, 'Gaussian Radial Basis Function')
            create_contour_plot(
                svm, X, y, 'Gaussian Radial Basis Function', filled=True)

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
        ax.set_title("$\mathbf{Support\ Vector\ Machine\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
