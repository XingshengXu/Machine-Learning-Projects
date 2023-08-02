import numpy as np


class SLPClassifier:
    """
    A Single Layer Perceptron (SLP) Classifier consists of a single layer of output nodes connected to 
    a layer of input nodes. The learning process involves adjusting the model parameters iteratively 
    to minimize the cross-entropy loss between the predicted and actual class labels.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        max_iterations (int, optional): Maximum number of iterations for the training loop.

    Attributes:
        iteration (int): Counter for iterations during training.
        cost (float): Current cost (cross-entropy loss) value.
        cost_memo (list): Record of cost at each iteration.
        X (np.array): Transposed input data matrix used for training.
        y (np.array): One-hot encoded actual class labels.
        theta (np.array): Model weights.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, learning_rate=0.1, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def softmax(self, x):
        """Build softmax function for converting network output to probability distribution."""

        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

        # Determine number of classes
        classe_number = len(np.unique(y))

        # Store the input data
        self.X = X.T

        # Convert labels to one-hot encoding
        self.y = np.eye(classe_number)[y].T

        # Determine sample number and feature number from input
        feature_number, sample_number = self.X.shape

        # Initialize weights theta
        self.theta = np.random.randn(feature_number, classe_number)

        # Single layer perceptron training
        while self.iteration < self.max_iterations:

            # Forward propagation: compute network output
            net = self.theta.T @ self.X
            pred_y = self.softmax(net)

            # Backward propagation: compute error and update weights
            error = self.y - pred_y
            grad = self.X @ error.T
            self.theta += self.learning_rate * grad

            # Compute the cross-entropy loss
            self.cost = -np.sum(self.y * np.log(pred_y + 1e-8)) / sample_number
            self.cost_memo.append(self.cost)

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")

        self.IsFitted = True

    def predict_class(self, X):
        """Predicts the label of given data using the trained model."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            h_func = self.theta.T @ X.T
            pred_y = self.softmax(h_func)

            # Convert the predicted outputs to label
            pred_label = np.argmax(pred_y, axis=0)
            return pred_label
