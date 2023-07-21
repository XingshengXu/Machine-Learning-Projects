import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(test_y, pred_y):
    """
    Evaluate the performance of a model by printing the Classification Report
    and creating Confusion Matrix.
    """

    # Print Classification Report
    print(classification_report(test_y, pred_y, zero_division=0))

    # Create Confusion Matrix
    cm = confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_cost_vs_iteration(model):
    """Plot cost vs. iteration."""

    plt.figure()
    plt.plot(range(1, model.iteration + 1), model.cost_memo)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.show()
