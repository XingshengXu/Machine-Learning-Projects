import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import path_setup
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Button
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from Decisiontree import ClassificationTree, RegressionTree
from Ensemble_Learning import *
from K_Nearest_Neighbors import KNNClassifier, KNNRegressor
from Logistic_Regression import LogisticRegression
from LWLR import LocalWeightedLinearRegression
from Single_Layer_Perceptron import SLPClassifier


def evaluate_model(test_y, pred_y, model_type):
    """
    Evaluate the performance of a model and prints the appropriate performance metrics 
    based on the type of the model. For classification models, it prints the classification 
    report and plots the confusion matrix. For regression models, it calculates and prints 
    the R-squared score.
    """

    if model_type == 'regression':

        # Calculate R-squared
        r2 = r2_score(test_y, pred_y)

        print("R-squared score (R²): ", r2)

    elif model_type == 'classification':

        # Print classification report
        print(classification_report(test_y, pred_y, zero_division=0))

        # Create confusion matrix
        cm = confusion_matrix(test_y, pred_y)
        sns.heatmap(cm, annot=True, fmt='d')

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    else:
        print("Invalid model type. Please choose either 'regression' or 'classification'.")


def create_contour_plot(model, X, y, resolution=500, alpha=0.5):
    """Create a contour plot for the decision boundaries of the trained ensemble model."""

    # Define the axis boundaries of the plot and create a meshgrid
    X_one_min, X_one_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    X_two_min, X_two_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Calculate the decision boundary of the model.
    X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, resolution),
                         np.linspace(X_two_min, X_two_max, resolution))
    X_plot = np.array([[x1, x2] for x1, x2 in zip(X1.flatten(), X2.flatten())])

    Y = model.predict_class(X_plot)
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
    plt.title('Decision Boundaries Visualized by Ensemble Model')

    plt.show()


def create_classification_animation(model, X, y, resolution=200, alpha=0.5):

    # Define the axis boundaries of the plot and create a meshgrid
    X_one_min, X_one_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    X_two_min, X_two_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, resolution),
                         np.linspace(X_two_min, X_two_max, resolution))

    # Create the plotting figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Define colormap
    cmap = plt.get_cmap('plasma', len(np.unique(y)))

    # Normalize data into [0.0, 1.0] range
    norm = mpl.colors.Normalize(vmin=np.min(y), vmax=np.max(y))

    def plot_decision_boundary(model_idx):
        # To start, consider only models up to the given index
        models_sublist = model.models[:model_idx+1]
        influence_sublist = model.influence[:model_idx+1]

        Z = np.zeros(X1.ravel().shape[0])

        # Calculate prediction for every point in the meshgrid
        for m, inf in zip(models_sublist, influence_sublist):
            Z += m.predict_class(np.c_[X1.ravel(), X2.ravel()]) * inf

        Z = (Z >= 0.5).reshape(X1.shape)

        ax.contourf(X1, X2, Z, alpha=alpha, cmap=cmap, norm=norm)
        for label in np.unique(y):
            points = X[y == label]
            color = cmap(norm(label))
            ax.scatter(points[:, 0], points[:, 1], color=color,
                       label=str(label), edgecolors='black')
        ax.set_title(
            f'Decision Boundary Evolution at AdaBoost Iteration {model_idx + 1}')

    def init():
        ax.clear()

    def update(frame):
        ax.clear()
        plot_decision_boundary(frame)

    ani = FuncAnimation(fig, update, frames=range(
        len(model.models)), init_func=init, repeat=False)
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')
    plt.legend(title="Class Labels")

    # Save the animation as a GIF
    writer = PillowWriter(fps=10)
    ani.save("classification_animation.gif", writer=writer)

    plt.show()


def find_grid_dimensions(n):
    """Find the number of rows and columns for a grid of n subplots."""

    for i in range(int(n**0.5), 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n


def create_subplots_random_forest(model, X, y, resolution=200, alpha=0.5):
    """
    Create a grid of subplots, each displaying the decision boundaries for a different 
    decision tree from a random forest classifier.
    """

    n = model.model_number
    rows, cols = find_grid_dimensions(n)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    X_one_min, X_one_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    X_two_min, X_two_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, resolution),
                         np.linspace(X_two_min, X_two_max, resolution))

    for i, ax in enumerate(axes[:n]):
        X_plot = np.array([[x1, x2]
                          for x1, x2 in zip(X1.flatten(), X2.flatten())])

        # Use base model to predict
        Y = model.base_models[i].predict_class(X_plot)
        Y = Y.reshape(X1.shape)

        cmap = plt.get_cmap('plasma', len(np.unique(y)))
        norm = mpl.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
        contour = ax.contourf(X1, X2, Y, alpha=alpha, cmap=cmap, norm=norm)

        for label in np.unique(y):
            points = X[y == label]
            color = cmap(norm(label))
            ax.scatter(points[:, 0], points[:, 1], color=color,
                       label=str(label), edgecolors='black')

        ax.set_title(f'Sub-Model {i+1} Plot')

    plt.tight_layout()
    plt.show()


def create_regression_plot(model, X, y):
    """Create a plot to visualize the ensemble model predictions."""

    # Get the predictions of Regression Tree
    pred_y = model.predict_value(X)

    # Plot the predictions of each model on the same set of axes.
    plt.figure(figsize=(10, 7))

    plt.scatter(X, y, s=20, c='black', marker='x', label='Target')
    plt.plot(sorted(X), [pred for _, pred in sorted(
        zip(X, pred_y))], c='red', label='Regression')

    plt.legend()
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')
    plt.title('Ensemble Model Regression Demo')
    plt.show()


def create_regression_animation(model, X, y):

    X = np.array(X)

    # If X is 1D, reshape it to 2D
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X, y, s=20, c='black', marker='x', label='Target')
    line, = ax.plot([], [], c='red', label='Regression')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        if frame == 0:
            pred = np.full(X.shape[0], model.models[0])
        else:
            # Start with the initial prediction
            pred = np.full(X.shape[0], model.models[0])

            # Sum up the predictions of each tree model up to the current frame
            for tree in model.models[1:frame+1]:
                # Check if the object is an instance of RegressionTree
                if isinstance(tree, RegressionTree):
                    pred += model.learning_rate * tree.predict_value(X)

        sorted_X, sorted_pred = zip(*sorted(zip(X, pred)))
        line.set_data(sorted_X, sorted_pred)
        ax.set_title(
            f'Ensemble Regression Evolution at Gradient Boosting Iteration {frame + 1}')
        return line,

    ani = FuncAnimation(fig, update, frames=range(
        len(model.models) + 1), init_func=init, blit=True, repeat=False)
    plt.legend()
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')

    # Save the animation as a GIF
    writer = PillowWriter(fps=10)
    ani.save("regression_animation.gif", writer=writer)

    plt.show()


def interactive_data_collection_classification(model_type):
    """
    Create an interactive plot for collecting data points for a Classification Task.

    The function allows you to interactively add points of two classes by left-clicking
    for class 0 and right-clicking for class 1. It also allows you to train an Ensemble 
    Model and visualize the decision boundaries by clicking the 'Train' button, or to 
    clear all data points and start over by clicking the 'Clean' button.
    """

    # Initialize click coordinates and labels
    coords, labels = [], []

    # Set color for each class
    class_colors = ['blue', 'yellow']

    # Create base estimators.
    slp = SLPClassifier()
    log_reg = LogisticRegression()
    knn = KNNClassifier()

    # Select the corresponding ensemble model
    if model_type == 'voting':
        model = VotingClassifier(
            classifiers=[slp, log_reg, knn], voting='hard')
    elif model_type == 'stacking':
        model = StackingClassifier(classifiers=[slp, knn])
    elif model_type == 'bagging':
        model = BaggingClassifier(
            base_model=ClassificationTree(), model_type='classifier', model_number=10)
    elif model_type == 'randomforest':
        model = RandomForestClassifier()
    elif model_type == 'adaboost':
        model = AdaBoostClassifier()
    else:
        print("Invalid model name. Please choose among"
              "\n'voting', 'stacking', 'bagging', and 'randomforest'.")

    # Create an interactive plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set plot properties
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("$\mathbf{Ensemble\ Model\ Classification\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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
            model.fit(X, y)
            create_contour_plot(model, X, y, resolution=200)

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
        ax.set_title("$\mathbf{Ensemble\ Model\ Classification\ Demo\ on\ Manually\ Generated\ Two-Class\ Data}$"
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


def interactive_data_collection_regression(model_type):
    """
    Create an interactive plot for collecting data points for a Regression Task.

    The function allows you to interactively add sample points by left-clicking. 
    It also allows you to train a Regression Tree and visualize the regression 
    plots by clicking the 'Train' button, or to clear all data points and start 
    over by clicking the 'Clear' button.
    """

    # Initialize click coordinates and values
    data_points = []

    # Create base estimators.
    tree = RegressionTree()
    knn = KNNRegressor()
    lwlr = LocalWeightedLinearRegression(tau=0.5)

    # Select the corresponding ensemble model
    if model_type == 'voting':
        model = VotingRegressor(regressors=[tree, knn, lwlr])
    elif model_type == 'stacking':
        model = StackingRegressor(regressors=[lwlr, knn])
    elif model_type == 'bagging':
        model = BaggingRegressor(
            base_model=RegressionTree(), model_type='regressor', model_number=10)
    elif model_type == 'randomforest':
        model = RandomForestRegressor()
    elif model_type == 'gradientboosting':
        model = GradientBoostingRegressor(model_number=50)
    else:
        print("Invalid model name. Please choose among"
              "\n'voting', 'stacking', 'bagging', and 'randomforest'.")

    # Create an interactive plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Set plot properties
    ax.set_xlim([-5, 5])
    ax.set_ylim([-50, 50])
    ax.set_title("$\mathbf{Ensemble\ Model\ Regression\ Demo\ on\ Manually\ Generated\ Data}$"
                 "\nLeft click to input data.")

    # Define onclick function for collecting data
    def onclick(event):
        if event.inaxes == ax:
            data_points.append((event.xdata, event.ydata))
            ax.scatter(event.xdata, event.ydata, s=20, c='black', marker='x')
            fig.canvas.draw()

    # Define onpress function for training model
    def onpress(event):
        if len(data_points) > 1:
            # reshaping to meet the input requirement
            X = np.array([dp[0] for dp in data_points]).reshape(-1, 1)
            y = np.array([dp[1] for dp in data_points])
            model.fit(X, y)
            create_regression_plot(model, X, y)

    # Define onclean function for resetting data
    def onclean(event):
        data_points.clear()

        # Remove drawn elements
        for coll in (ax.collections + ax.lines):
            coll.remove()
        fig.canvas.draw()

        ax.set_xlim([-5, 5])
        ax.set_ylim([-50, 50])
        ax.set_title("$\mathbf{Ensemble\ Model\ Regression\ Demo\ on\ Manually\ Generated\ Data}$"
                     "\nLeft click to input data.")
        fig.canvas.draw()

    # Create 'Train' and 'Clear' buttons
    ax_button_train = plt.axes([0.25, 0.01, 0.2, 0.06])
    button_train = Button(ax_button_train, 'Train')
    button_train.on_clicked(onpress)

    ax_button_clear = plt.axes([0.55, 0.01, 0.2, 0.06])
    button_clear = Button(ax_button_clear, 'Clear')
    button_clear.on_clicked(onclean)

    # Register onclick function
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
