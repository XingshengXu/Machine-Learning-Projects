"""
Radial Basis Function Networks for Handwritten Digit Recognition
"""

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def softmax(x):
    """Softmax Function"""
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


def gaussian(x, center, sigma):
    """Gaussian Radial Basis Function"""
    return np.exp(-0.5 * np.sum(((x - center) / sigma) ** 2, axis=0))


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
cluster_number = 20
max_iterations = 3000
max_kmeans_iterations = 100
train_set_size = len(train_set)
test_set_size = len(test_set)
weights_output = np.random.randn(cluster_number, 10)
weights_output_prev = np.zeros(weights_output.shape)

# Initializations
cost = 0
cost_memo = []
iteration = 0
dist = np.zeros((cluster_number, train_set_size))
std_devs = np.zeros(cluster_number)

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
    train_set_size, cluster_number, replace=False)]

for kmeans_iterations in range(max_kmeans_iterations):
    cluster_centers_prev = cluster_centers.copy()

    # Assignment Step: Assign each example to the closest center
    for i in range(cluster_number):
        dist[i, :] = np.linalg.norm(
            X_train - cluster_centers[:, i, np.newaxis], axis=0)

    class_id = np.argmin(dist, axis=0)

    # Update Step: Compute new centers as the mean of all examples assigned to each cluster
    for i in range(cluster_number):
        points_in_cluster = X_train[:, class_id == i]

        # Check If the Cluster Has Any Points
        if points_in_cluster.size > 0:
            cluster_centers[:, i] = np.mean(points_in_cluster, axis=1)
        else:
            # If No Points, Reinitialize the Center Randomly
            cluster_centers[:, i] = X_train[:,
                                            np.random.choice(train_set_size, 1)]
    print(f'kmeans_iterations: {kmeans_iterations}')

# Compute the Standard Deviation for Each Cluster
for i in range(cluster_number):
    points_in_cluster = X_train[:, class_id == i]

    # Check If the Cluster Has Any Points
    if points_in_cluster.size > 0:
        # Compute the Mean Squared Distance to the Cluster Center
        squared_distances = np.linalg.norm(
            points_in_cluster - cluster_centers[:, i, np.newaxis], axis=0) ** 2
        mean_squared_distance = np.mean(squared_distances)

        # Compute the Standard Deviation for Each Cluster
        std_devs[i] = np.sqrt(mean_squared_distance)
    else:
        # If No Points, Assign Some Default Value
        std_devs[i] = 1.0

# Compute the Cluster Nodes
cluster_nodes = np.zeros((cluster_number, X_train.shape[1]))
for i in range(cluster_number):
    cluster_nodes[i, :] = gaussian(
        X_train, cluster_centers[:, i, np.newaxis], std_devs[i])

# Radial Basis Function Networks Training
while iteration <= max_iterations:
    net_output = weights_output.T @ cluster_nodes
    Y_pred = softmax(net_output)

    # Compute the Cross-Entropy Loss
    cost = -np.sum(Y_train * np.log(Y_pred + 1e-8)) / train_set_size
    cost_memo.append(cost)

    # Backpropagation
    delta_output = Y_train - Y_pred

    # Update Output Weights
    grad_output = cluster_nodes @ delta_output.T
    weights_output_prev = alpha_momentum * weights_output_prev + \
        learning_rate * grad_output / train_set_size
    weights_output += weights_output_prev

    iteration += 1
    print(f'{iteration} iterations')

print(f"Training finished after {iteration} iterations.")

# Plot Learning Curve
plt.plot(range(1, iteration + 1), cost_memo)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# Multi Layer Perceptron Testing

# Compute the Cluster Nodes in Test Set
cluster_nodes_test = np.zeros((cluster_number, X_test.shape[1]))

for i in range(cluster_number):
    cluster_nodes_test[i, :] = gaussian(
        X_test, cluster_centers[:, i, np.newaxis], std_devs[i])

net_output_test = weights_output.T @ cluster_nodes_test
Y_pred_test = softmax(net_output_test)

# Convert the Predicted Outputs to Label Form
Y_pred_test_label = np.argmax(Y_pred_test, axis=0)

# Print Classification Report
print(classification_report(test_label, Y_pred_test_label, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(test_label, Y_pred_test_label)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
