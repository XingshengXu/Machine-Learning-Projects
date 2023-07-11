import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.widgets import Button
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from decisiontree import ClassificationTree


def evaluate_model(test_Y, pred_Y):
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


def create_contour_plot(tree, X, y, resolution=500, alpha=0.5):
    """Create a contour plot for the decision boundaries of the trained Decision Tree."""

    # Define the axis boundaries of the plot and create a meshgrid
    X_one_min, X_one_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    X_two_min, X_two_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Calculate the decision boundary of the model.
    X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, resolution),
                         np.linspace(X_two_min, X_two_max, resolution))
    X_plot = [[x1, x2] for x1, x2 in zip(X1.flatten(), X2.flatten())]

    Y = np.array([tree.predict_class(np.array(x)) for x in X_plot])
    Y = Y.reshape(X1.shape)

    plt.figure(figsize=(10, 7))

    # Define colormap
    cmap = plt.get_cmap('plasma', len(np.unique(y)))

    # Normalize data into [0.0, 1.0] range
    norm = mpl.colors.Normalize(vmin=np.min(y), vmax=np.max(y))

    # Plot the decision boundary
    contour = plt.contourf(X1, X2, Y, alpha=alpha, cmap=cmap, norm=norm)

    # Plot the training data for each class with a legend
    for label in np.unique(y):
        points = X[y == label]
        color = cmap(norm(label))
        plt.scatter(points[:, 0], points[:, 1], color=color,
                    label=str(label), edgecolors='black')

    plt.legend(title="Class Labels")
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')
    plt.title('Decision Boundaries Visualized by Classification Tree')

    plt.show()


def interactive_data_collection_classification():
    """
    Create an interactive plot for collecting data points for a Classification Task.

    The function allows you to interactively add points of two classes by left-clicking
    for class 0 and right-clicking for class 1. It also allows you to train a classification 
    tree and visualize the decision boundaries by clicking the 'Train' button, or to clear 
    all data points and start over by clicking the 'Clean' button.
    """

    # Initialize click coordinates and labels
    coords, labels = [], []

    # Set color for each class
    class_colors = ['blue', 'yellow']

    # Create an interactive plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set plot properties
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("$\mathbf{Classification\ Tree\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
            tree = ClassificationTree()
            tree.fit(X, y)
            create_contour_plot(tree, X, y)

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
        ax.set_title("$\mathbf{Classification\ Tree\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
