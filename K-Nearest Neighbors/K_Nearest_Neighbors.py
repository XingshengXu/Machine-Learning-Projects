import numpy as np


class KNN:
    """ 
    This is a super class for KNNClassifier and KNNRegressor. It encapsulates 
    functionality for finding 'k' nearest neighbours in a feature space.

    Args:
        k_neighbours (int): The number of nearest neighbours to consider 
                            when calculating the target value of an instance.

    Attributes:
        X (numpy.ndarray): A matrix containing the feature values of the training set.
        y (numpy.ndarray): A vector containing the target values of the training set.
        neighbour_idx (numpy.ndarray): Indices of the k nearest neighbours in the training set.
        k_nearest_neighbours (numpy.ndarray): The k nearest neighbours of a given sample.
        neighbour_values (numpy.ndarray): The labels or targets of the k nearest neighbours.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, k_neighbours=5):
        self.k_neighbours = k_neighbours
        self.IsFitted = False

    def find_k_nearest_neighbours(self, sample):
        """Find the k-nearest neighbours of the given sample point."""

        dists = np.linalg.norm(self.X - sample, axis=1)
        self.neighbour_idx = np.argpartition(dists, self.k_neighbours)[
            :self.k_neighbours]
        self.k_nearest_neighbours = self.X[self.neighbour_idx]
        self.neighbour_values = self.y[self.neighbour_idx]

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

        if len(X.shape) == 1:
            self.X = np.array(X).reshape(-1, 1)
        else:
            self.X = np.array(X)
        self.y = np.array(y)

        self.IsFitted = True


class KNNClassifier(KNN):
    """
    The K-Nearest Neighbors (KNN) Classifier is a type of instance-based learning algorithm, 
    used for classification tasks. It classifies a new instance based on the majority label 
    of its 'k' nearest neighbors in the feature space.
    """

    def predict_proba(self, sample):
        """Predict the class probabilities of the given sample point."""

        self.find_k_nearest_neighbours(sample)
        counts = np.bincount(self.neighbour_values)
        class_probabilities = counts / np.sum(counts)

        return class_probabilities

    def predict_class(self, X):
        """Predict the class of the given instance."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            return np.array([np.argmax(self.predict_proba(x)) for x in X])


class KNNRegressor(KNN):
    """
    The K-Nearest Neighbors (KNN) Regressor is an instance-based learning algorithm used for 
    regression tasks. It predicts the value for a new instance as the average value of its 'k' 
    nearest neighbors.
    """

    def _predict_value(self, sample):
        """
        Private method to predict the target value for a single instance using the 
        k-nearest neighbors.
        """

        self.find_k_nearest_neighbours(sample)
        return np.mean(self.neighbour_values)

    def predict_value(self, X):
        """Predict the target value of the given instance."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            return np.array([self._predict_value(x) for x in X])
