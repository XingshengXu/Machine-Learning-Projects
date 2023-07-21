import numpy as np


class SLPClassifier:
    """
    A Single Layer Perceptron Classifier consists of a single layer of output nodes connected to 
    a layer of input nodes. The learning process involves adjusting the model parameters iteratively 
    to minimize the cross-entropy loss between the predicted and actual class labels.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        max_iterations (int, optional): Maximum number of iterations for the training loop.

    Attributes:
        sigma (float): Threshold value for determining class labels from output.
        threshold (int): Threshold value for converting grayscale images to binary images.
        iteration (int): Counter for iterations during training.
        cost (float): Current cost (cross-entropy loss) value.
        cost_memo (list): Record of cost at each iteration.
        sample_number (int): Number of training samples.
        image_size (int): Size of each input image.
        theta (np.array): Model weights.
        image (np.array): Processed input images used for training.
        label (np.array): One-hot encoded actual class labels.
    """

    def __init__(self, learning_rate=0.1, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.sigma = 0
        self.threshold = 128
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def grayscale_to_binary(self, image, threshold):
        """Greyscale to Binary Image Function."""

        filter_image = image >= threshold
        return filter_image

    def fit(self, image, label):
        """Train the model with the given training set."""

        # Determine sample numbers and image_size from input
        self.sample_number, self.image_size = image.shape[0:2]

        # Initialize theta now that we know image_size
        self.theta = np.random.randn(self.image_size ** 2, 10)

        # Transfer images from grayscale to binary
        self.image = self.grayscale_to_binary(image, self.threshold)

        # Reshape image size from 2D to 1D
        self.image = self.image.reshape(self.sample_number, -1).T

        # Convert labels to expected outputs
        self.label = np.eye(10)[label].T

        # Single layer perceptron training
        while self.iteration <= self.max_iterations:
            net = self.theta.T @ self.image
            pred_y = net >= self.sigma
            error = self.label - pred_y
            grad = self.image @ error.T
            self.theta += self.learning_rate * grad

            # Compute the cross-entropy loss
            self.cost = -np.sum(self.label *
                                np.log(pred_y + 1e-8)) / self.sample_number
            self.cost_memo.append(self.cost)

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")
        print(f"Theta values:{self.theta}")

        self.IsFitted = True

    def predict_label(self, image):
        """Predicts the label of given images using the trained model."""

        # Convert grayscale images to binary images
        image = self.grayscale_to_binary(image, self.threshold)
        image = image.reshape(image.shape[0], -1).T

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            h_func = self.theta.T @ image
            pred_y = h_func >= self.sigma

            # Convert the predicted outputs to label
            pred_label = np.argmax(pred_y, axis=0)
            return pred_label
