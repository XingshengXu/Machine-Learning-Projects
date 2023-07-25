import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import multivariate_normal
from Gaussian_Discriminant_Analysis import GaussianDiscriminant


def evaluate_model(test_y, pred_y):
    """
    Evaluate the performance of a model by printing the Classification Report
    and creating Confusion Matrix.
    """

    # Print Classification Report
    print(classification_report(test_y, pred_y, zero_division=0))

    # Create Confusion Matrix
    cm = confusion_matrix(test_y, pred_y)

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_class_distributions(model):
    """
    Plot the Gaussian distributions of two classes and 
    their decision boundary, along with the training data.
    """

    # Create Grid and Multivariate Normal Distributions
    x = np.linspace(-3, 3, 500)
    y = np.linspace(-3, 3, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean Vector and Covariance Matrix for Class 0
    mu0 = np.squeeze(model.mean_y0)
    Sigma0 = model.covariance_matrix

    # Mean Vector and Covariance Matrix for Class 1
    mu1 = np.squeeze(model.mean_y1)
    Sigma1 = model.covariance_matrix

    # Create A Frozen RV Object for Each Class
    rv0 = multivariate_normal(mu0, Sigma0)
    rv1 = multivariate_normal(mu1, Sigma1)

    # Calculate the Probabilities for Each Class on the Grid
    prob_grid_y0 = model.prob_y0 * rv0.pdf(pos)
    prob_grid_y1 = model.prob_y1 * rv1.pdf(pos)

    # Create A Mask Where the Probability of Class 1 is Greater Than Class 0
    decision_boundary = prob_grid_y1 > prob_grid_y0

    # Make A Contour Plot for Each Class
    plt.figure(figsize=(10, 7))
    plt.contourf(X, Y, rv0.pdf(pos), alpha=0.7, cmap='Reds')
    plt.contourf(X, Y, rv1.pdf(pos), alpha=0.7, cmap='Blues')

    # Plot the Decision Boundary
    plt.contour(X, Y, decision_boundary, colors='green')

    # Define A Colormap
    cmap = ListedColormap(['red', 'blue'])

    # Plot the Data
    plt.scatter(model.X[0, :], model.X[1, :],
                c=model.y, cmap=cmap, edgecolors='k')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gaussian Distributions and Decision Boundary')
    plt.show()


def interactive_data_collection_classification():
    """
    Create an interactive plot for collecting data points for a Classification Task.

    The function allows you to interactively add points of two classes by left-clicking
    for class 0 and right-clicking for class 1. It also allows you to train a Gaussian 
    Discriminant Analysis Classifier and visualize the decision boundary by clicking 
    the 'Train' button, or to clear all data points and start over by clicking the 'Clean' 
    button.
    """

    # Initialize click coordinates and labels
    coords, labels = [], []

    # Set color for each class
    class_colors = ['red', 'blue']

    # Create an interactive plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set plot properties
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("$\mathbf{Gaussian\ Discriminant\ Classifier\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
            gda = GaussianDiscriminant()
            gda.fit(X, y)
            plot_class_distributions(gda)

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
        ax.set_title("$\mathbf{Gaussian\ Discriminant\ Classifier\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
