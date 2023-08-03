import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, r2_score


def preprocess_image(image):
    """Preprocesses the input image for training."""

    # Determine sample numbers from input
    sample_number = image.shape[0]

    # Mean Normalization
    image = (image - np.mean(image)) / np.std(image)

    # Reshape image size from 3D to 2D (sample_number, sample_size*sample_size)
    X = image.reshape(sample_number, -1)

    return X


def evaluate_model(test_y, pred_y, model_type):
    """
    Evaluate the performance of a model and prints the appropriate performance metrics 
    based on the type of the model. For classification models, it prints the classification 
    report and plots the confusion matrix. For regression models, it calculates and prints 
    the R-squared score.
    """

    if model_type == 'regression':

        # Ensure pred_y is 1D
        pred_y = pred_y.flatten()

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


def plot_cost_vs_iteration(model):
    """Plot cost vs. iteration."""

    plt.figure()
    plt.plot(range(1, model.iteration + 1), model.cost_memo)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.show()


def create_contour_plot(model, X, y, resolution=500, alpha=0.5):
    """Create a contour plot for the decision boundaries of the trained model."""

    # Define the axis boundaries of the plot and create a meshgrid
    X_one_min, X_one_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    X_two_min, X_two_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Calculate the decision boundary of the model.
    X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, resolution),
                         np.linspace(X_two_min, X_two_max, resolution))
    X_plot = np.array([[x1, x2] for x1, x2 in zip(X1.flatten(), X2.flatten())])

    Y = np.array(model.predict_class(X_plot))
    Y = Y.reshape(X1.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(X1, X2, Y, alpha=alpha, cmap='jet')

    # Plot the training data, color-coded based on their true label
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='jet')

    plt.xlabel('Feature one')
    plt.ylabel('Feature two')
    plt.title('Decision Boundaries Visualized by RBF Networks')
    plt.show()


def create_regression_plot(model, X, y):
    """Create a plot to visualize the regression predictions."""

    X = np.array(X)
    y = np.array(y)

    # Get the predictions of Regression Tree
    pred_y = model.predict_value(X)

    # Ensure pred_y is 2D
    pred_y = pred_y.T

    # Plot the predictions of each model on the same set of axes.
    plt.figure(figsize=(10, 7))

    plt.scatter(X, y, s=20, c='black', marker='x', label='Target')
    plt.plot(sorted(X), [pred for _, pred in sorted(
        zip(X, pred_y))], c='red', label='Regression')

    plt.legend(loc='upper center')
    plt.xlabel('Feature one')
    plt.ylabel('Feature two')
    plt.title('RBF Networks Regression Demo')
    plt.show()
