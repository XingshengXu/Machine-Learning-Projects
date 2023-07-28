import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix


class AdaptiveResonanceTheory():
    """
    Implementation of the Fuzzy Adaptive Resonance Theory (ART) clustering algorithm.
    ART is a form of unsupervised learning. It is designed to quickly categorize input patterns 
    and maintain the stability of recognized clusters in response to changing inputs. 
    The term "adaptive resonance" refers to the system's capacity to self-adjust (adapt) 
    and create a "resonance", or state of harmony, when a new or familiar input signal is presented.

    Args:
        learning_rate (float, optional): The learning rate of the algorithm which determines how fast 
                                         the weights are updated. Default is 0.1.
        alpha (int, optional): The choice parameter, contributes to the denominator in the choice function 
                               to prevent very small or very large input vectors from dominating the learning 
                               process. Default is 1.
        epsilon (float, optional): The vigilance parameter, determines how strict the model is when assigning 
                                   input vectors to existing clusters. Default is 0.6.

    Attributes:
        iteration (int): Current iteration count.
        IsEqual (bool): Boolean flag to check whether cluster assignments have changed.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
        weights (numpy.ndarray): Weight matrix where each column represents the weight vector for a cluster.
        cluster_id (numpy.ndarray): Array that holds the cluster id for each instance in the training set.
        valid_indices (list): List of indices for clusters that have at least one associated instance.
        num_valid_clusters (list): A list to hold the number of valid clusters during the model fitting.
    """

    def __init__(self, learning_rate=0.1, alpha=1, epsilon=0.6):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epsilon = epsilon
        self.iteration = 0
        self.IsEqual = False
        self.IsFitted = False
        self.cluster_changes = []
        self.num_valid_clusters = []

    def choice_function(self, X_train):
        """Computes the choice function which ranks clusters in order of fitness to be selected."""

        #! T_j = Σ min(X_i, w_ji) / (α + Σ w_ji)
        min_matrix = np.minimum(self.weights, X_train.T)
        choice_score = min_matrix.sum(
            axis=1) / (self.alpha + self.weights.sum(axis=1))
        return np.argsort(choice_score)[::-1]

    def vigilance_test(self, X_train, candidate_index):
        """Computes the vigilance test which verifies if the selected cluster matches 
        closely enough with the input vector."""

        #! S_j = Σ min(X_i, w_ji) / Σ X_i

        min_matrix = np.minimum(self.weights[candidate_index, :], X_train)
        match_score = np.sum(min_matrix) / np.sum(X_train)

        if match_score >= self.epsilon:  # If match score meets the vigilance criterion

            #! Update weights of the neuron: w_ji_new = β * min(w_ji_old, X_i) + (1-β) * w_ji_old
            self.weights[candidate_index, :] = self.learning_rate * min_matrix + (
                1 - self.learning_rate) * self.weights[candidate_index, :]
            return True
        return False

    def count_valid_clusters(self):
        """Count the number of unique elements in cluster_id. This gives the number 
        of clusters that have at least one instance associated with them"""

        unique_clusters = np.unique(self.cluster_id)
        valid_cluster_count = len(unique_clusters)
        return valid_cluster_count

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (n_samples, n_features).
        """

        self.cluster_id = np.zeros(X.shape[0], dtype=np.int32)
        self.weights = np.array([X[0, :]])

        while not self.IsEqual:
            cluster_id_prev = self.cluster_id.copy()

            for i in range(X.shape[0]):
                # Compute choice function
                candidate_indices = self.choice_function(X[i, :])

                # Vigilance test
                for candidate_index in candidate_indices:
                    if self.vigilance_test(X[i, :], candidate_index):
                        self.cluster_id[i] = candidate_index
                        break
                else:
                    self.weights = np.vstack((self.weights, X[i, :]))
                    self.cluster_id[i] = self.weights.shape[0] - 1

            self.IsEqual = np.array_equal(cluster_id_prev, self.cluster_id)
            self.iteration += 1
            self.num_valid_clusters.append(self.count_valid_clusters())
            changes = np.count_nonzero(self.cluster_id != cluster_id_prev)
            self.cluster_changes.append(changes)
            print(
                f'Iteration {self.iteration}: {np.count_nonzero(self.cluster_id != cluster_id_prev)} different clusters')

        # Compute labels for each cluster
        self.cluster_labels = [np.argmax(np.bincount(y[self.cluster_id == id_value])) if np.any(
            self.cluster_id == id_value) else None for id_value in range(self.weights.shape[0])]

        # The model has been fitted
        self.IsFitted = True

    def predict(self, X):
        '''Return the model's predicted cluster for each of the given instances.'''

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            predicted_labels = [self.predict_label(
                test_data, self.cluster_labels) for test_data in X]
        return predicted_labels

    def predict_label(self, X_test, cluster_labels):
        """Predicts the label of an instance by finding the cluster with the highest choice function value."""

        min_matrix = np.minimum(self.weights, X_test)
        choice_score = min_matrix.sum(
            axis=1) / (self.alpha + self.weights.sum(axis=1))

        # Only consider clusters that have a label
        self.valid_indices = [i for i in range(
            len(cluster_labels)) if cluster_labels[i] is not None]
        valid_scores = choice_score[self.valid_indices]

        # Find the cluster with the highest choice function value
        best_match_index = self.valid_indices[np.argmax(valid_scores)]

        # Assign the label of the best match cluster to the test example
        return cluster_labels[best_match_index]

    def predict_test(self, test_Y, pred_Y):
        """This function prints the classification report and plots the confusion 
        matrix for the given actual and predicted labels."""

        # Print classification report
        print(classification_report(test_Y, pred_Y, zero_division=0))

        # Create confusion matrix
        cm = confusion_matrix(test_Y, pred_Y)
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        # Plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 7))

        # Plot the number of changes among clusters after each iteration
        axs[0].plot(range(self.iteration), self.cluster_changes,
                    color='red', alpha=0.7, linewidth=2.5)
        axs[0].set(xlabel='Iteration', ylabel='Changes Among Clusters')
        axs[0].set_title('Changes Among Clusters over Iterations')

        # Plot the number of valid clusters after each iteration
        axs[1].plot(range(self.iteration), self.num_valid_clusters,
                    color='blue', alpha=0.7, linewidth=2.5)
        axs[1].set(xlabel='Iteration', ylabel='Number of Valid Clusters')
        axs[1].set_title('Number of Valid Clusters over Iterations')

        # Adjust spacing between subplots
        fig.tight_layout()
        plt.show()
