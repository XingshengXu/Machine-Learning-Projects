import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def evaluate_model(model, test_y, pred_y):
    """Evaluate the performance of a model and prints the R-squared score."""

    # Calculate R-squared
    r2 = r2_score(test_y, pred_y)

    print(f"Training finished after {model.iteration} iterations.")
    print(
        f"Theta values: theta_0 = {model.theta_real[0]}, theta_1 = {model.theta_real[1]}")
    print("R-squared score (R²): ", r2)


def plot_cost_vs_iteration(model):
    """Plot cost vs. iteration."""

    plt.figure()
    plt.plot(range(1, model.iteration + 1), model.cost_memo)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.show()
