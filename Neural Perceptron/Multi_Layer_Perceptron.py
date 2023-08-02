import numpy as np


class MLPClassifier:
    """
    A Multi-Layer Perceptron (MLP) classifier with one hidden layer.
    The MLP is trained using a simple form of batch gradient descent, with a sigmoid activation 
    function for the hidden layer and a softmax function for the output layer. The cost function is 
    cross-entropy, and a momentum term is included to improve convergence.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        max_iterations (int, optional): Maximum number of iterations for the training loop.
        hidden_node_number (int, optional): Number of nodes in the hidden layer.

    Attributes:
        alpha_momentum (float): Momentum term used to accelerate convergence.
        iteration (int): Counter for tracking the number of iterations during training.
        cost (float): Current cost or loss (cross-entropy loss) value.
        cost_memo (list): A list storing the cost at each iteration for tracking the performance of the model.
        weights_hidden (np.array): Weights of the hidden layer in the model.
        weights_output (np.array): Weights of the output layer in the model.
        weights_hidden_prev (np.array): Weights update from the previous iteration for the hidden layer.
        weights_output_prev (np.array): Weights update from the previous iteration for the output layer.
        X (np.array): Transposed input data matrix used for training.
        y (np.array): One-hot encoded actual class labels.
        IsFitted (bool): Boolean flag to indicate if the model has been trained.
    """

    def __init__(self, learning_rate=0.4, max_iterations=1000, hidden_node_number=10):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.hidden_node_number = hidden_node_number
        self.alpha_momentum = 0.9
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def sigmoid(self, x):
        """Build sigmoid function."""

        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Build softmax function for converting network output to probability distribution."""

        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

        # Determine sample number and feature number from input
        sample_number, feature_number = X.shape

        # Determine number of classes
        classe_number = len(np.unique(y))

        # Initialize weights
        self.weights_hidden = np.random.randn(
            feature_number, self.hidden_node_number)
        self.weights_output = np.random.randn(
            self.hidden_node_number, classe_number)
        self.weights_hidden_prev = np.zeros(self.weights_hidden.shape)
        self.weights_output_prev = np.zeros(self.weights_output.shape)

        # Store the input data
        self.X = X.T

        # Convert labels to one-hot encoding
        self.y = np.eye(classe_number)[y].T

        # Multi layer perceptron training
        while self.iteration < self.max_iterations:
            net_hidden = self.weights_hidden.T @ self.X
            hidden_nodes = self.sigmoid(net_hidden)
            net_output = self.weights_output.T @ hidden_nodes
            pred_y = self.softmax(net_output)

            # Compute the Cross-Entropy Loss
            self.cost = -np.sum(self.y *
                                np.log(pred_y + 1e-8)) / sample_number
            self.cost_memo.append(self.cost)

            # Backpropagation
            delta_output = self.y - pred_y
            delta_hidden = (self.weights_output @ delta_output) * \
                hidden_nodes * (1 - hidden_nodes)

            # Update Output Weights
            grad_output = hidden_nodes @ delta_output.T
            self.weights_output_prev = self.alpha_momentum * self.weights_output_prev + \
                self.learning_rate * grad_output / sample_number
            self.weights_output += self.weights_output_prev

            # Update Hidden Weights
            grad_hidden = self.X @ delta_hidden.T
            self.weights_hidden_prev = self.alpha_momentum * self.weights_hidden_prev + \
                self.learning_rate * grad_hidden / sample_number
            self.weights_hidden += self.weights_hidden_prev

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")

        self.IsFitted = True

    def predict_class(self, X):
        """Predicts the label of given data using the trained model."""

        X = X.T

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            net_hidden = self.weights_hidden.T @ X
            hidden_nodes = self.sigmoid(net_hidden)
            net_output = self.weights_output.T @ hidden_nodes
            pred_y = self.softmax(net_output)

            # Convert the Predicted Outputs to Label Form
            pred_label = np.argmax(pred_y, axis=0)
            return pred_label
