import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def generate_test_data(n_samples=100, n_features=2, std=0.5):
    '''Generate a test dataset with n-dimensional instances.'''
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2,
                      random_state=0, cluster_std=std)

    # Normalization of input data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X).T

    # Complement coding of input data to prevent the weight decreases too fast
    X = np.vstack((X, 1.0 - X))
    return X, y


def complement_coding(input_data):
    """This function calculates the complement of a given input data 
    and stacks it horizontally with the original input data."""
    # Ensure the input is a numpy array.
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    # Calculate the complement of the input matrix.
    complement_data = 1.0 - input_data

    # Stack the original input matrix and its complement horizontally.
    return np.hstack((input_data, complement_data))


def predict_test(test_Y, pred_Y):
    """This function prints the classification report and plots the confusion 
    matrix for the given actual and predicted labels."""
    # Print Classification Report
    print(classification_report(test_Y, pred_Y, zero_division=0))

    # Create Confusion Matrix
    cm = confusion_matrix(test_Y, pred_Y)

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


class AdaptiveResonanceTheory():
    def __init__(self, learning_rate=0.1, alpha=10, epsilon=0.6):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.iteration = 0
        self.IsEqual = False
        self.IsFitted = False

    def choice_function(self, X_train, weights):
        min_matrix = np.minimum(weights, X_train)
        choice_score = np.sum(min_matrix, axis=0) / (
            self.alpha + np.sum(weights, axis=0))
        return np.argsort(choice_score)[::-1]

    def vigilance_test(self, X_train, weights, candidate_index):
        min_matrix = np.minimum(weights[:, candidate_index], X_train)
        match_score = np.sum(min_matrix) / np.sum(X_train)

        if match_score >= self.epsilon:
            weights[:, candidate_index] = self.learning_rate * min_matrix + (
                1 - self.learning_rate) * weights[:, candidate_index]
            return True
        return False

    def predict_label(self, X_test, weights, cluster_labels):
        X_test = X_test.reshape(-1, 1)
        min_matrix = np.minimum(weights, X_test)
        choice_score = np.sum(min_matrix, axis=0) / (
            self.alpha + np.sum(weights, axis=0))

        # Only consider clusters that have a label
        self.valid_indices = [i for i in range(
            len(cluster_labels)) if cluster_labels[i] is not None]
        valid_scores = choice_score[self.valid_indices]

        # Find the category with the highest choice function value
        best_match_index = self.valid_indices[np.argmax(valid_scores)]

        # Assign the label of the best match category to the test example
        return cluster_labels[best_match_index]

    def fit(self, X, y):

        self.cluster_id = np.zeros(X.shape[1], dtype=np.int32)
        self.weights = X[:, 0].reshape(-1, 1)

        while not self.IsEqual:
            cluster_id_prev = self.cluster_id.copy()

            for i in range(X.shape[1]):
                # Compute choice function
                candidate_indices = self.choice_function(
                    X[:, [i]], self.weights)

                # Vigilance test
                for candidate_index in candidate_indices:
                    if self.vigilance_test(X[:, i], self.weights, candidate_index):
                        self.cluster_id[i] = candidate_index
                        break
                else:
                    self.weights = np.column_stack((self.weights, X[:, i]))
                    self.cluster_id[i] = self.weights.shape[1] - 1

            self.IsEqual = np.array_equal(cluster_id_prev, self.cluster_id)
            self.iteration += 1
            print(
                f'Iteration {self.iteration}: {np.count_nonzero(self.cluster_id != cluster_id_prev)} different clusters')

        # Compute labels for each cluster
        self.cluster_labels = [np.argmax(np.bincount(y[self.cluster_id == id_value])) if np.any(
            self.cluster_id == id_value) else None for id_value in range(self.weights.shape[1])]

        # The model has been fitted
        self.IsFitted = True

    def predict(self, X):
        '''Return the models predicted class for each of the given instances.'''
        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            predicted_labels = [self.predict_label(
                test_data, self.weights, self.cluster_labels) for test_data in X.T]
        return predicted_labels
