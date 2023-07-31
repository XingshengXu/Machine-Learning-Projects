import numpy as np


class Node():
    """ Represents a single node in the Decision Tree.

    Args:
        feature_threshold (tuple): Stores the feature index and threshold value used for
                                   the decision at the node. Format: (feature index, threshold).
        class_frequency (numpy.ndarray): Frequency of classes at this node.
        prediction (float): The predicted class for the samples at this node in the case of 
                            classification or the predicted target value in the case of regression.

    Attributes:
        left (Node): The left child node of the current node.
        right (Node): The right child node of the current node.
    """

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
    """
    The super Tree class for the ClassificationTree class and the RegressionTree class.

    Args:
        max_depth (int, optional): Maximum depth for the tree.
        min_split_samples (int, optional): Minimum number of samples required to split a node.
        min_leaf_samples (int, optional): Minimum number of samples required to be at a leaf node.

    Attributes:
        root (Node): The root node of the Decision Tree.
    """

    def __init__(self, max_depth=np.inf, min_split_samples=2, min_leaf_samples=1):
        self.root = None

        # Check if the arguments are valid
        if max_depth > 0:
            self.max_depth = max_depth
        else:
            raise AttributeError(
                'Invalid value, max_depth must be greater than zero.')

        if min_leaf_samples > 0 and min_split_samples >= 2 * min_leaf_samples:
            self.min_leaf_samples = min_leaf_samples
            self.min_split_samples = min_split_samples
        else:
            raise AttributeError(
                'Invalid values: min_leaf_samples must be greater than zero, and'
                'min_split_samples must be no less than twice min_leaf_samples.'
            )

    def split(self, split_column, split_threshold, X, y):
        """Split the dataset based on the given feature and threshold value."""

        left_mask = X[:, split_column] <= split_threshold
        right_mask = X[:, split_column] > split_threshold

        # The separated samples and corresponding labels
        left = X[left_mask]
        y_left = y[left_mask].astype(int)

        right = X[right_mask]
        y_right = y[right_mask].astype(int)

        return left, y_left, right, y_right

    def __predict_probs(self, subtree, sample):
        """Private method to return the class frequencies of the sample associated with the node."""

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

    def predict_proba(self, sample):
        """Predict the class probabilities of the sample for Ensemble Models."""

        if self.root is None:
            raise AttributeError(
                f"Model not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            classes = self.__predict_proba(self.root, sample)
            return classes / np.sum(classes)

    def predict_value(self, sample):
        """Predict the output value of the sample based on average value for Regression Tree."""

        if self.root is None:
            raise AttributeError(
                f"Model not fitted, call 'fit' with appropriate arguments before using model.")

        # Perform prediction traversing the entire tree
        value = self.__predict_probs(self.root, sample)
        return value

    def predict_class(self, sample):
        """Predict the class of the sample based on majority frequency for Classification Tree."""

        if self.root is None:
            raise AttributeError(
                f"Model not fitted, call 'fit' with appropriate arguments before using model.")

        # Perform prediction traversing the entire tree
        classes = self.__predict_probs(self.root, sample)
        return np.argmax(classes)


class ClassificationTree(Tree):
    """ A Classification Tree is a type of Decision Tree that involves top-down, greedy, and 
    recursive partitioning of the data space to create a tree-like model for classification tasks.

    Args:
        criterion (str, optional): The metric to determine the efficiency of a split decision.
        max_depth (int, optional): Maximum depth for the tree.
        min_split_samples (int, optional): Minimum number of samples required to split a node.
        min_leaf_samples (int, optional): Minimum number of samples required to be at a leaf node.

    Attributes:
        root (Node): The root node of Classification Tree.
    """

    def __init__(self, criterion='gini', max_depth=np.inf, min_split_samples=2, min_leaf_samples=1):

        super().__init__(max_depth, min_split_samples, min_leaf_samples)

        # Check if the criterion is valid
        if criterion == 'gini' or criterion == 'entropy':
            self.criterion = criterion
        else:
            raise AttributeError(
                "Invalid criterion, 'gini' and 'entropy' are available.")

    def gini_impurity(self, y):
        """Calculate the Gini impurity of the specified node."""

        _, counts = np.unique(y, return_counts=True)
        total = np.sum(counts)
        probabilities = counts / total
        #! Gini = 1 - Σ (p_i)^2
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def entropy(self, y):
        """Calculate the Entropy of the specified node."""

        _, counts = np.unique(y, return_counts=True)
        total = np.sum(counts)
        probabilities = counts / total
        #! Entropy = - Σ p_i * log2(p_i)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def information_gain(self, y_left, y_right, y_parent):
        """Calculate the information gain of the specified split."""

        n = y_parent.shape[0]
        entropy_parent = self.entropy(y_parent)

        #! Entropy_Total = Σ ( |Di| / |D| ) * Entropy(Di)
        total_entropy = ((y_left.shape[0] / n) * self.entropy(y_left) +
                         (y_right.shape[0] / n) * self.entropy(y_right))

        info_gain = entropy_parent - total_entropy
        return info_gain

    def calculate_improvement(self, y_left, y_right, y_parent):
        """Calculate the appropriate impurity value based on the criterion
        being used in the model."""

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
        """Calculates the split points in the dataset. Returns the list 
        of these split points and the corresponding feature indices."""

        feature_indices = []
        split_points = []

        # Calculates the split points by averaging each pair of adjacent unique values in each feature.
        for i, x in enumerate(np.sort(X.T)):
            x = np.unique(x)
            for j in range(x.shape[0]-1):
                split = (x[j] + x[j+1])/2.0
                split_points.append(split)
                feature_indices.append(i)
        return feature_indices, split_points

    def best_split(self, X, y):
        """Determine the best split for a Classification Tree based on the specified impurity criterion.
        The function will return the split that results in the largest decrease in impurity."""

        # Get possible split points
        feature_indices, split_points = self.get_split_points(X)

        # Initialize best impurity measure

        if self.criterion == 'gini':
            best_impurity = np.inf
        else:
            best_impurity = -1

        # Initialize best left, right, and split_point as None
        best_left, best_right, best_split_point = None, None, None

        # Evaluate the impurity of each possible split point
        for feature_index, split_point in zip(feature_indices, split_points):
            left, y_left, right, y_right = self.split(
                feature_index, split_point, X, y)
            current_impurity = self.calculate_improvement(y_left, y_right, y)

            # Update the best split that results in the largest decrease in impurity
            is_better_gini_split = self.criterion == 'gini' and current_impurity < best_impurity
            is_better_entropy_split = self.criterion == 'entropy' and current_impurity > best_impurity

            if is_better_gini_split or is_better_entropy_split:
                best_left = np.c_[left, y_left]
                best_right = np.c_[right, y_right]
                best_impurity = current_impurity
                best_split_point = (feature_index, split_point)

        return best_left, best_right, best_split_point

    def __fit(self, subtree, curr_depth, X, y):
        """Private method to recursively build the Classification Tree until all instances have been 
        correctly classified or certain criteria have been satisfied."""

        # Count the frequency of each class in node
        class_frequency = np.bincount(y.astype(int))

        # Base case for recursion: If any of the regulation criteria are met, return a leaf node
        is_gini_zero = (
            self.criterion == 'gini' and self.gini_impurity(y) == 0)
        is_entropy_zero = (
            self.criterion == 'entropy' and self.entropy(y) == 0)

        if (
            curr_depth > self.max_depth or
            len(X) < self.min_split_samples or
            is_gini_zero or is_entropy_zero
        ):
            return Node(feature_threshold=None, class_frequency=class_frequency)
        else:
            # Recursive case: If not a leaf node, compute the best split
            (best_left, best_right, best_split_point) = self.best_split(X, y)

            # If no valid split point is found, return a leaf node
            if best_split_point is None:
                return Node(feature_threshold=None, class_frequency=class_frequency)

            # Check if the split leads to leaf nodes with too few samples
            if (len(best_left) < self.min_leaf_samples or
                    len(best_right) < self.min_leaf_samples):
                return Node(feature_threshold=None, class_frequency=class_frequency)
            else:
                subtree = Node(best_split_point, class_frequency)
                X_left, y_left = best_left[:, :-1], best_left[:, -1]
                X_right, y_right = best_right[:, :-1], best_right[:, -1]
                subtree.left = self.__fit(
                    subtree.left, curr_depth + 1, X_left, y_left)
                subtree.right = self.__fit(
                    subtree.right, curr_depth + 1, X_right, y_right)
                return subtree

    def fit(self, X, y):
        '''
        Build the Classification Tree by recursively calling the __fit method.
        Note, the input data matrix should have the shape of (n_samples, n_features).
        '''
        self.root = self.__fit(self.root, 1, X, y)


class RegressionTree(Tree):
    """ A Regression Tree is a type of Decision Tree that involves top-down, greedy, and 
    recursive partitioning of the data space to create a tree-like model for regression tasks.

    Args:
        criterion (str, optional): The metric to determine the efficiency of a split decision.
        max_depth (int, optional): Maximum depth for the tree.
        min_split_samples (int, optional): Minimum number of samples required to split a node.
        min_leaf_samples (int, optional): Minimum number of samples required to be at a leaf node.

    Attributes:
        root (Node): The root node of Regression Tree.
    """

    def __init__(self, criterion='mse', max_depth=np.inf, min_split_samples=2, min_leaf_samples=1):
        super().__init__(max_depth, min_split_samples, min_leaf_samples)

        # Check if the criterion is valid
        if criterion == 'mse' or criterion == 'mae':
            self.criterion = criterion
        else:
            raise AttributeError(
                "Invalid criterion, 'mse' and 'mae' are available.")

    def MSE(self, y):
        '''Calculate the mean squared error of the samples at the node per feature.'''
        y_hat = np.mean(y)
        mse = np.sum((y_hat - y)**2)
        return mse

    def MAE(self, y):
        """Calculate the mean absolute error of the samples at the node per feature."""
        y_hat = np.mean(y)
        mae = np.sum(np.abs(y_hat - y))
        return mae

    def calculate_improvement(self, y_left, y_right, y_parent):
        """Calculate the reduction in error based on the criterion
        being used in the model."""

        n = y_parent.shape[0]

        if self.criterion == 'mse':
            #! MSE_Total = Σ ( |Di| / |D| ) * MSE(Di)
            total_mse = (y_left.shape[0] / n) * self.MSE(y_left) + (
                y_right.shape[0] / n) * self.MSE(y_right)
            return total_mse
        else:
            #! MAE_Total = Σ ( |Di| / |D| ) * MAE(Di)
            total_mae = (y_left.shape[0] / n) * self.MAE(y_left) + (
                y_right.shape[0] / n) * self.MAE(y_right)
            return total_mae

    def get_split_points(self, X):
        """Calculates the split points in the dataset. Returns the list 
        of these split points and the corresponding feature indices."""

        feature_indices = []
        split_points = []

        # Calculates the split points by averaging each pair of adjacent unique values in each feature.
        for i, x in enumerate(np.sort(X.T)):
            x = np.unique(x)
            for j in range(x.shape[0]-1):
                split = (x[j] + x[j+1])/2.0
                split_points.append(split)
                feature_indices.append(i)
        return feature_indices, split_points

    def best_split(self, X, y):
        """Determine the best split for a Regression Tree based on the specified error criterion.
        The function will return the split that results in the largest decrease in error."""

        if X.ndim == 1:
            X = X.reshape((-1, 1))

        # Get possible split points
        feature_indices, split_points = self.get_split_points(X)

        # Initialize lowest error measure
        lowest_error = np.inf

        # Initialize best left, right, and split_point as None
        best_left, best_right, best_split_point = None, None, None

        # Evaluate the error of each possible split point
        for feature_index, split_point in zip(feature_indices, split_points):
            left, y_left, right, y_right = self.split(
                feature_index, split_point, X, y)
            current_error = self.calculate_improvement(y_left, y_right, y)

            # Update the best split if current error is the lowest one
            if current_error < lowest_error:
                best_left = np.c_[left, y_left]
                best_right = np.c_[right, y_right]
                lowest_error = current_error
                best_split_point = (feature_index, split_point)

        return best_left, best_right, best_split_point

    def __fit(self, subtree, curr_depth, X, y):
        """Private method to recursively build the Regression Tree until all instances have been 
        correctly regressed or certain criteria have been satisfied."""

        # Calculate the mean value of node
        mean_y = np.mean(y)

        # Base case for recursion: If any of the regulation criteria are met, return a leaf node
        is_mse_zero = (self.criterion == 'mse' and self.MSE(y) == 0)
        is_mae_zero = (self.criterion == 'mae' and self.MAE(y) == 0)

        if (
            curr_depth > self.max_depth or
            len(X) < self.min_split_samples or
            is_mse_zero or is_mae_zero
        ):
            return Node(feature_threshold=None, class_frequency=mean_y)
        else:
            # Recursive case: If not a leaf node, compute the best split
            (best_left, best_right, best_split_point) = self.best_split(X, y)

            # If no valid split point is found, return a leaf node
            if best_split_point is None:
                return Node(feature_threshold=None, class_frequency=mean_y)

            # Check if the split leads to leaf nodes with too few samples
            if (len(best_left) < self.min_leaf_samples or
                    len(best_right) < self.min_leaf_samples):
                return Node(feature_threshold=None, class_frequency=mean_y)
            else:
                subtree = Node(best_split_point, mean_y)
                X_left, y_left = best_left[:, :-1], best_left[:, -1]
                X_right, y_right = best_right[:, :-1], best_right[:, -1]
                subtree.left = self.__fit(
                    subtree.left, curr_depth + 1, X_left, y_left)
                subtree.right = self.__fit(
                    subtree.right, curr_depth + 1, X_right, y_right)
                return subtree

    def fit(self, X, y):
        '''Build the Regression Tree by recursively calling the __fit method.'''
        self.root = self.__fit(self.root, 1, X, y)
