import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error


class Node():
    def __init__(self, feature_threshold, class_frequency=None, prediction=None):
        self.feature_threshold = feature_threshold

        # Classification Tree
        if prediction is None:
            self.class_frequency = class_frequency
            self.prediction = np.argmax(class_frequency)
        # Regression Tree
        else:
            self.prediction = prediction
        self.left = None
        self.right = None


class Tree():
    def __init__(self, max_depth=np.inf, min_split_samples=2, min_leaf_samples=1):
        self.root = None

        if max_depth > 0:
            self.max_depth = max_depth
        else:
            raise AttributeError(
                'Invalid max_depth value, max_depth must be greater than zero.')

        if min_leaf_samples > 0 and min_split_samples > 2 * min_leaf_samples:
            self.min_leaf_samples = min_leaf_samples
            self.min_split_samples = min_split_samples
        else:
            raise AttributeError(
                'Invalid values: min_samples_leaf must be greater than zero, and min_samples_split must be no less than twice min_samples_leaf.')

    def split(self, split_threshold, split_column, X, y):
        '''Split the dataset based on the given feature and threshold value.'''

        left_mask = X[:, split_column] <= split_threshold
        right_mask = X[:, split_column] > split_threshold

        # The separated samples and corresponding labels
        left = X[left_mask]
        y_left = y[left_mask].astype(int)

        right = X[right_mask]
        y_right = y[right_mask].astype(int)

        return left, y_left, right, y_right

    def __predict_probs(self, subtree, sample):
        '''Private function to return the class frequencies of the sample associated with the node.'''

        # Reshape the input if the sample is a scalar
        if sample.ndim == 0:
            sample = np.array([sample])

        # If the subtree is a leaf, return the class frequencies
        if subtree.feature_threshold is None:
            return subtree.class_frequency

        # If the value is less than or equal to the split threshold, predict on left subtree
        if sample[subtree.feature_threshold[0]] <= subtree.feature_threshold[1]:
            return self.__predict_probs(subtree.left, sample)

        # Otherwise, predict on the right subtree
        return self.__predict_probs(subtree.right, sample)

    def predict_probs(self, sample):
        '''Predict the class probabilities of the sample for regression.'''

        if self.root is None:
            raise AttributeError(
                f"Model not fitted, call 'fit' with appropriate arguments before using model.")

        # Perform prediction traversing the entire tree
        classes = self.__predict_probs(self.root, sample)
        classes /= np.sum(classes)
        return classes

    def predict_class(self, sample):
        '''Predict the class of the sample based on frequencies for classification.'''

        if self.root is None:
            raise AttributeError(
                f"Model not fitted, call 'fit' with appropriate arguments before using model.")

        # Perform prediction traversing the entire tree
        classes = self.__predict_probs(self.root, sample)
        return np.argmax(classes)


class ClassificationTree(Tree):
    def __init__(self, criterion='gini', max_depth=np.inf, min_split_samples=2, min_leaf_samples=1):

        super().__init__(max_depth, min_split_samples, min_leaf_samples)

        if criterion == 'gini' or criterion == 'entropy':
            self.criterion = criterion
        else:
            raise AttributeError(
                f"Invalid criterion, 'gini' and 'entropy' are available.")

    def gini_impurity(self, y):
        '''Calculate the Gini impurity of the specified node.'''

        _, counts = np.unique(y, return_counts=True)
        total = np.sum(counts)
        probabilities = counts / total
        #! Gini = 1 - Σ (p_i)^2
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def entropy(self, y):
        '''Calculate the Entropy of the specified node.'''

        _, counts = np.unique(y, return_counts=True)
        total = np.sum(counts)
        probabilities = counts / total
        #! Entropy = - Σ p_i * log2(p_i)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, y_left, y_right, y_parent):
        '''Calculate the information gain of the specified split.'''

        n = y_parent.shape[0]
        entropy_parent = self.entropy(y_parent)

        #! Entropy_Total = Σ ( |Di| / |D| ) * Entropy(Di)
        total_entropy = ((y_left.shape[0] / n) * self.entropy(y_left) +
                         (y_right.shape[0] / n) * self.entropy(y_right))

        info_gain = entropy_parent - total_entropy
        return info_gain

    def calculate_improvement(self, y_left, y_right, y_parent):
        '''Calculate the appropriate impurity value based on the criterion
        being used in the model.'''

        n = y_parent.shape[0]
        if self.criterion == 'gini':
            gini_left = self.gini_impurity(y_left)
            gini_right = self.gini_impurity(y_right)

            #! Gini_Total = Σ ( |Di| / |D| ) * Gini(Di)
            total_gini = (y_left.shape[0] / n) * gini_left + (
                y_right.shape[0] / n) * gini_right
            return total_gini
        else:
            info_gain = self.information_gain(y_left, y_right, y_parent)
            return info_gain

    def get_split_points(self, X):
        '''Calculates the split points in the dataset. Returns the list of these split points and the corresponding feature indices.'''

        feature_indices = []
        split_points = []

        # Calculates the split points by averaging each pair of adjacent unique values in each feature.
        for i, x in enumerate(np.sort(X.T)):
            x = np.unique(x)
            for j in range(x.shape[0]-1):
                split = (x[j] + x[j+1])/2.0
                split_points.append(split)
                feature_indices.append(i)
        return split_points, feature_indices

    def best_split(self, X, y):
        '''Determine the best split for a decision tree based on the specified impurity criterion.
        The function will return the split that results in the largest decrease in impurity.'''

        # Get possible split points
        split_points, feature_indices = self.get_split_points(X)

        # Initialize best impurity measure

        if self.criterion == 'gini':
            best_impurity = np.inf
        else:
            best_impurity = -1

        # Evaluate each possible split point
        for feature_index, split_value in zip(feature_indices, split_points):

            lower, y_lower, upper, y_upper = self.split(
                split_value, feature_index, X, y)
            current_impurity = self.calculate_improvement(y_lower, y_upper, y)

            # Update if this split is better than the best so far
            is_better_gini_split = self.criterion == 'gini' and current_impurity < best_impurity
            is_better_entropy_split = self.criterion == 'entropy' and current_impurity > best_impurity
            if is_better_gini_split or is_better_entropy_split:
                best_left = np.c_[lower, y_lower]
                best_right = np.c_[upper, y_upper]
                best_impurity = current_impurity
                best_split_point = (split_value, feature_index)

        return best_left, best_right, best_split_point
