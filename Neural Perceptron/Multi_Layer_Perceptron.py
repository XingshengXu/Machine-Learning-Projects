import numpy as np


class MLPClassifier:
    """
    A Multi-Layer Perceptron (MLP) classifier with one hidden layer.
    The MLP is trained using a simple form of stochastic gradient descent, with a sigmoid activation 
    function for the hidden layer and a softmax function for the output layer. The cost function is 
    cross-entropy, and a momentum term is included to improve convergence.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        max_iterations (int, optional): Maximum number of iterations for the training loop.
        hidden_node_number (int, optional): Number of nodes in the hidden layer.

    Attributes:
        learning_rate (float): The learning rate for gradient descent.
        max_iterations (int): The maximum number of iterations for the training loop.
        hidden_node_number (int): The number of nodes in the hidden layer.
        alpha_momentum (float): Momentum term to improve convergence.
        iteration (int): Counter for iterations during training.
        cost (float): Current cost (cross-entropy loss) value.
        cost_memo (list): Record of cost at each iteration.
        sample_number (int): Number of training samples.
        image_size (int): Size of each input image.
        weights_hidden (np.array): Model weights for the hidden layer.
        weights_output (np.array): Model weights for the output layer.
        weights_hidden_prev (np.array): Previous update for weights of the hidden layer.
        weights_output_prev (np.array): Previous update for weights of the output layer.
        image (np.array): Processed input images used for training.
        label (np.array): One-hot encoded actual class labels.
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
        """Build softmax function."""

        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    def fit(self, image, label):
        """Train the model with the given training set."""

        # Determine sample numbers and image_size from input
        self.sample_number, self.image_size = image.shape[0:2]

        # Initialize theta now that we know image_size
        self.weights_hidden = np.random.randn(
            self.image_size ** 2, self.hidden_node_number)
        self.weights_output = np.random.randn(self.hidden_node_number, 10)
        self.weights_hidden_prev = np.zeros(self.weights_hidden.shape)
        self.weights_output_prev = np.zeros(self.weights_output.shape)

        # Transfer images from grayscale to binary
        self.image = image / 255

        # Reshape image size from 2D to 1D
        self.image = self.image.reshape(self.sample_number, -1).T

        # Convert labels to expected outputs
        self.label = np.eye(10)[label].T

        # Single layer perceptron training
        while self.iteration <= self.max_iterations:
            net_hidden = self.weights_hidden.T @ self.image
            hidden_nodes = self.sigmoid(net_hidden)
            net_output = self.weights_output.T @ hidden_nodes
            pred_y = self.softmax(net_output)

            # Compute the Cross-Entropy Loss
            self.cost = -np.sum(self.label *
                                np.log(pred_y + 1e-8)) / self.sample_number
            self.cost_memo.append(self.cost)

            # Backpropagation
            delta_output = self.label - pred_y
            delta_hidden = (self.weights_output @ delta_output
                            ) * hidden_nodes * (1 - hidden_nodes)

            # Update Output Weights
            grad_output = hidden_nodes @ delta_output.T
            self.weights_output_prev = self.alpha_momentum * self.weights_output_prev + \
                self.learning_rate * grad_output / self.sample_number
            self.weights_output += self.weights_output_prev

            # Update Hidden Weights
            grad_hidden = self.image @ delta_hidden.T
            self.weights_hidden_prev = self.alpha_momentum * self.weights_hidden_prev + \
                self.learning_rate * grad_hidden / self.sample_number
            self.weights_hidden += self.weights_hidden_prev

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")

        self.IsFitted = True

    def predict_label(self, image):
        """Predicts the label of given images using the trained model."""

        # Convert grayscale images to binary images
        image = image / 255
        image = image.reshape(image.shape[0], -1).T

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            net_hidden = self.weights_hidden.T @ image
            hidden_nodes = self.sigmoid(net_hidden)
            net_output = self.weights_output.T @ hidden_nodes
            pred_y = self.softmax(net_output)

            # Convert the Predicted Outputs to Label Form
            pred_label = np.argmax(pred_y, axis=0)
            return pred_label
