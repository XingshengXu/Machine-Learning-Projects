"""
Multi Layer Perceptron Classifier for Handwritten Digit Recognition
"""

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def sigmoid(x):
    """Sigmoid Function"""
    return 1 / (1 + np.exp(-x))


def gaussian(x, center, sigma):
    """Gaussian Radial Basis Function"""
    return np.exp(-0.5 * ((x - center) / sigma) ** 2)


# Load Training and Testing Data Sets
try:
    train_set = idx2numpy.convert_from_file(
        './Perceptron/dataset/train-images.idx3-ubyte')
    train_label = idx2numpy.convert_from_file(
        './Perceptron/dataset/train-labels.idx1-ubyte')
    test_set = idx2numpy.convert_from_file(
        './Perceptron/dataset/t10k-images.idx3-ubyte')
    test_label = idx2numpy.convert_from_file(
        './Perceptron/dataset/t10k-labels.idx1-ubyte')
except FileNotFoundError as e:
    print("One or more data files not found.")
    print(e)
    exit()

# Parameter Settings
image_size = 28
learning_rate = 0.4
alpha_momentum = 0.9
RBF_number = 20
max_iterations = 100
max_kmeans_iterations = 10
train_set_size = len(train_set)
test_set_size = len(test_set)
weights_output = np.random.randn(RBF_number, 10)
weights_output_prev = np.zeros(weights_output.shape)

# Initializations
cost = 0
cost_memo = []
iteration = 0
dist = np.zeros((RBF_number, train_set_size))
std_devs = np.zeros(RBF_number)

# Normalize Input Images
train_images = train_set / 255
test_images = test_set / 255

# Reshape Image Size From 2D to 1D
X_train = train_images.reshape(train_set_size, -1).T
X_test = test_images.reshape(test_set_size, -1).T

# Convert Labels to Expected Outputs
Y_train = np.eye(10)[train_label].T

# K-means Clustering
np.random.seed(42)
cluster_centers = X_train[:, np.random.choice(
    train_set_size, RBF_number, replace=False)]

for kmeans_iterations in range(max_kmeans_iterations):
    cluster_centers_prev = cluster_centers.copy()

    # Assignment step: Assign each example to the closest center
    for i in range(RBF_number):
        dist[i, :] = np.linalg.norm(
            X_train - cluster_centers[:, i, np.newaxis], axis=0)

    class_id = np.argmin(dist, axis=0)

    # Update step: Compute new centers as the mean of all examples assigned to each cluster
    for i in range(RBF_number):
        cluster_centers[:, i] = np.mean(X_train[:, class_id == i], axis=1)

# Compute the standard deviation for each cluster
for i in range(RBF_number):
    # Only consider data points assigned to the current cluster
    points_in_cluster = X_train[:, class_id == i]

    # Compute the mean squared distance to the cluster center
    squared_distances = np.linalg.norm(
        points_in_cluster - cluster_centers[:, i, np.newaxis], axis=0) ** 2
    mean_squared_distance = np.mean(squared_distances)

    # Compute the standard deviation
    std_devs[i] = np.sqrt(mean_squared_distance)
