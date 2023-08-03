import numpy as np


class RBFClassifier:
    """
    The Radial Basis Function (RBF) Networks for classification. The RBF network is a type 
    of artificial neural network that uses radial basis functions as activation functions. 
    It uses KMeans algorithm to select centers and applies gaussian radial basis function.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        cluster_number (int, optional): Number of clusters for the KMeans algorithm.
        max_iterations (int, optional): Maximum number of iterations for the training loop.

    Attributes:
        alpha_momentum (float): The momentum coefficient used in the weight update rule.
        max_kmeans_iterations (int): The maximum number of iterations for the KMeans algorithm.
        iteration (int): Counter for iterations during training.
        cost (float): Current cost (cross-entropy loss) value.
        cost_memo (list): Record of cost at each iteration.
        sample_number (int): Number of training samples.
        weights_output (np.array): Model weights.
        weights_output_prev (np.array): Previous model weights.
        X (np.array): Transposed input data used for training.
        y (np.array): Transposed one-hot encoded actual class labels.
        cluster_nodes (np.array): The nodes in each cluster.
        cluster_centers (np.array): The centers of each cluster.
        std_devs (np.array): The standard deviations of each cluster.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, learning_rate=0.4, cluster_number=20, max_iterations=3000):
        self.learning_rate = learning_rate
        self.cluster_number = cluster_number
        self.max_iterations = max_iterations
        self.alpha_momentum = 0.9
        self.max_kmeans_iterations = 100
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def softmax(self, x):
        """Build softmax function."""

        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    def gaussian(self, x, center, sigma):
        """Build gaussian radial basis function as activation function."""

        return np.exp(-0.5 * np.sum(((x - center) / sigma) ** 2, axis=0))

    def kmeans(self, X):
        """
        Applies the KMeans algorithm to the provided data to find cluster centers and standard deviations.
        Returns the cluster nodes, cluster centers, and standard deviations for each cluster.
        """

        # Initiate cluster distance
        dist = np.zeros((self.cluster_number, self.sample_number))

        # Initiate standard deviations
        std_devs = np.zeros(self.cluster_number)

        # Initiate cluster centers
        np.random.seed(42)
        cluster_centers = X[:, np.random.choice(
            self.sample_number, self.cluster_number, replace=False)]

        for kmeans_iterations in range(self.max_kmeans_iterations):

            # Assignment Step: Assign each example to the closest center
            for i in range(self.cluster_number):
                dist[i, :] = np.linalg.norm(
                    X - cluster_centers[:, i, np.newaxis], axis=0)

            class_id = np.argmin(dist, axis=0)

            # Update Step: Compute new centers as the mean of all examples assigned to each cluster
            for i in range(self.cluster_number):
                points_in_cluster = X[:, class_id == i]

                if points_in_cluster.size > 0:
                    cluster_centers[:, i] = np.mean(points_in_cluster, axis=1)
                else:
                    cluster_centers[:, i] = X[:,
                                              np.random.choice(self.sample_number, 1)]
            print(f'kmeans_iterations: {kmeans_iterations}')

        # Compute the standard deviation for each cluster
        for i in range(self.cluster_number):
            points_in_cluster = X[:, class_id == i]

            if points_in_cluster.size > 0:
                squared_distances = np.linalg.norm(
                    points_in_cluster - cluster_centers[:, i, np.newaxis], axis=0) ** 2
                mean_squared_distance = np.mean(squared_distances)
                std_devs[i] = np.sqrt(mean_squared_distance)
            else:
                std_devs[i] = 1.0

        # Compute the cluster nodes
        cluster_nodes = np.zeros((self.cluster_number, self.X.shape[1]))
        for i in range(self.cluster_number):
            cluster_nodes[i, :] = self.gaussian(
                self.X, cluster_centers[:, i, np.newaxis], std_devs[i])

        return cluster_nodes, cluster_centers, std_devs

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

        # Determine sample number
        self.sample_number = X.shape[0]

        # Initialize weights
        self.weights_output = np.random.randn(
            self.cluster_number, len(np.unique(y)))
        self.weights_output_prev = np.zeros(self.weights_output.shape)

        # Store the input data
        self.X = X.T

        # Convert labels to one-hot encoding
        self.y = np.eye(len(np.unique(y)))[y].T

        # Perform KMeans clustering to initialize the RBF network
        self.cluster_nodes, self.cluster_centers, self.std_devs = self.kmeans(
            self.X)

        # Radial basis function networks training
        while self.iteration < self.max_iterations:
            net_output = self.weights_output.T @ self.cluster_nodes
            pred_y = self.softmax(net_output)

            # Compute the cross-entropy loss
            self.cost = -np.sum(self.y *
                                np.log(pred_y + 1e-8)) / self.sample_number
            self.cost_memo.append(self.cost)

            # Backpropagation
            delta_output = self.y - pred_y

            # Update output weights
            grad_output = self.cluster_nodes @ delta_output.T
            self.weights_output_prev = self.alpha_momentum * self.weights_output_prev + \
                self.learning_rate * grad_output / self.sample_number
            self.weights_output += self.weights_output_prev

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")

        self.IsFitted = True

    def predict_class(self, X):
        """Predicts the label of given data using the trained model."""

        X = X.T

        # Initialize the cluster nodes
        cluster_nodes = np.zeros((self.cluster_number, X.shape[1]))

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            for i in range(self.cluster_number):
                cluster_nodes[i, :] = self.gaussian(
                    X, self.cluster_centers[:, i, np.newaxis], self.std_devs[i])

            net_output = self.weights_output.T @ cluster_nodes
            pred_y = self.softmax(net_output)

            # Convert the predicted outputs to label
            pred_label = np.argmax(pred_y, axis=0)
            return pred_label


class RBFRegressor:
    """
    The Radial Basis Function (RBF) Networks for regression tasks. The RBF network is a type 
    of artificial neural network that uses radial basis functions as activation functions. 
    It uses the KMeans algorithm to select centers and applies the gaussian radial basis function.

    Args:
        learning_rate (float, optional): Learning rate for updating model parameters.
        cluster_number (int, optional): Number of clusters for the KMeans algorithm.
        max_iterations (int, optional): Maximum number of iterations for the training loop.

    Attributes:
        alpha_momentum (float): The momentum coefficient used in the weight update rule.
        max_kmeans_iterations (int): The maximum number of iterations for the KMeans algorithm.
        iteration (int): Counter for iterations during training.
        cost (float): Current cost (mean squared error) value.
        cost_memo (list): Record of cost at each iteration.
        sample_number (int): Number of training samples.
        weights_output (np.array): Model weights for output layer.
        weights_output_prev (np.array): Previous model weights for output layer.
        X (np.array): Transposed input data used for training.
        y (np.array): Transposed target values.
        cluster_nodes (np.array): The nodes in each cluster.
        cluster_centers (np.array): The centers of each cluster.
        std_devs (np.array): The standard deviations of each cluster.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, learning_rate=0.4, cluster_number=20, max_iterations=3000):
        self.learning_rate = learning_rate
        self.cluster_number = cluster_number
        self.max_iterations = max_iterations
        self.alpha_momentum = 0.9
        self.max_kmeans_iterations = 100
        self.iteration = 0
        self.cost = 0
        self.cost_memo = []
        self.IsFitted = False

    def gaussian(self, x, center, sigma):
        """Build gaussian radial basis function as activation function."""

        return np.exp(-0.5 * np.sum(((x - center) / sigma) ** 2, axis=0))

    def kmeans(self, X):
        """
        Applies the KMeans algorithm to the provided data to find cluster centers and standard deviations.
        Returns the cluster nodes, cluster centers, and standard deviations for each cluster.
        """

        # Initiate cluster distance
        dist = np.zeros((self.cluster_number, self.sample_number))

        # Initiate standard deviations
        std_devs = np.zeros(self.cluster_number)

        # Initiate cluster centers
        np.random.seed(42)
        cluster_centers = X[:, np.random.choice(
            self.sample_number, self.cluster_number, replace=False)]

        for kmeans_iterations in range(self.max_kmeans_iterations):

            # Assignment Step: Assign each example to the closest center
            for i in range(self.cluster_number):
                dist[i, :] = np.linalg.norm(
                    X - cluster_centers[:, i, np.newaxis], axis=0)

            class_id = np.argmin(dist, axis=0)

            # Update Step: Compute new centers as the mean of all examples assigned to each cluster
            for i in range(self.cluster_number):
                points_in_cluster = X[:, class_id == i]

                if points_in_cluster.size > 0:
                    cluster_centers[:, i] = np.mean(points_in_cluster, axis=1)
                else:
                    cluster_centers[:, i] = X[:,
                                              np.random.choice(self.sample_number, 1)]
            print(f'kmeans_iterations: {kmeans_iterations}')

        # Compute the standard deviation for each cluster
        for i in range(self.cluster_number):
            points_in_cluster = X[:, class_id == i]

            if points_in_cluster.size > 0:
                squared_distances = np.linalg.norm(
                    points_in_cluster - cluster_centers[:, i, np.newaxis], axis=0) ** 2
                mean_squared_distance = np.mean(squared_distances)
                std_devs[i] = np.sqrt(mean_squared_distance)
            else:
                std_devs[i] = 1.0

        # Compute the cluster nodes
        cluster_nodes = np.zeros((self.cluster_number, self.X.shape[1]))
        for i in range(self.cluster_number):
            cluster_nodes[i, :] = self.gaussian(
                self.X, cluster_centers[:, i, np.newaxis], std_devs[i])

        return cluster_nodes, cluster_centers, std_devs

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

        # Determine sample number
        self.sample_number = X.shape[0]

        # Initialize weights
        self.weights_output = np.random.randn(self.cluster_number, 1)
        self.weights_output_prev = np.zeros(self.weights_output.shape)

        # Store the input data
        self.X = X.T
        self.y = y.reshape(-1, 1).T

        # Perform KMeans clustering to initialize the RBF network
        self.cluster_nodes, self.cluster_centers, self.std_devs = self.kmeans(
            self.X)

        # Radial basis function networks training
        while self.iteration < self.max_iterations:
            net_output = self.weights_output.T @ self.cluster_nodes
            pred_y = net_output

            # Compute the mean square error
            self.cost = np.mean(np.square(self.y - pred_y)
                                ) / self.sample_number
            self.cost_memo.append(self.cost)

            # Backpropagation
            delta_output = self.y - pred_y

            # Update output weights
            grad_output = self.cluster_nodes @ delta_output.T
            self.weights_output_prev = self.alpha_momentum * self.weights_output_prev + \
                self.learning_rate * grad_output / self.sample_number
            self.weights_output += self.weights_output_prev

            self.iteration += 1
            print(f'{self.iteration} iterations')

        print(f"Training finished after {self.iteration} iterations.")

        self.IsFitted = True

    def predict_value(self, X):
        """Predict the target value of the given instance."""

        X = X.T

        # Initialize the cluster nodes
        cluster_nodes = np.zeros((self.cluster_number, X.shape[1]))

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            for i in range(self.cluster_number):
                cluster_nodes[i, :] = self.gaussian(
                    X, self.cluster_centers[:, i, np.newaxis], self.std_devs[i])

            net_output = self.weights_output.T @ cluster_nodes
            pred_y = net_output

            return pred_y
