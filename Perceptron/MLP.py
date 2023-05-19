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


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


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
hidden_node_number = 20
max_iterations = 3000
train_set_size = len(train_set)
test_set_size = len(test_set)
weights_hidden = np.random.randn(image_size ** 2, hidden_node_number)
weights_output = np.random.randn(hidden_node_number, 10)
weights_hidden_prev = np.zeros(weights_hidden.shape)
weights_output_prev = np.zeros(weights_output.shape)

# Initializations
cost = 0
cost_memo = []
iteration = 0

# Normalize Input Images
train_images = train_set / 255
test_images = test_set / 255

# Reshape Image Size From 2D to 1D
X_train = train_images.reshape(train_set_size, -1).T
X_test = test_images.reshape(test_set_size, -1).T

# Convert Labels to Expected Outputs
Y_train = np.eye(10)[train_label].T

# Multi Layer Perceptron Training
while iteration <= max_iterations:
    net_hidden = weights_hidden.T @ X_train
    hidden_nodes = sigmoid(net_hidden)
    net_output = weights_output.T @ hidden_nodes
    Y_pred = softmax(net_output)

    # Compute the Cross-Entropy Loss
    cost = -np.sum(Y_train * np.log(Y_pred + 1e-8)) / train_set_size
    cost_memo.append(cost)

    # Backpropagation
    delta_output = Y_train - Y_pred
    delta_hidden = (weights_output @ delta_output
                    ) * hidden_nodes * (1 - hidden_nodes)

    # Update Output Weights
    grad_output = hidden_nodes @ delta_output.T
    weights_output_prev = alpha_momentum * weights_output_prev + \
        learning_rate * grad_output / train_set_size
    weights_output += weights_output_prev

    # Update Hidden Weights
    grad_hidden = X_train @ delta_hidden.T
    weights_hidden_prev = alpha_momentum * weights_hidden_prev + \
        learning_rate * grad_hidden / train_set_size
    weights_hidden += weights_hidden_prev

    iteration += 1
    print(f'{iteration} iterations')

print(f"Training finished after {iteration} iterations.")

# Plot Learning Curve
plt.plot(range(1, iteration + 1), cost_memo)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# Single Layer Perceptron Testing
net_hidden_test = weights_hidden.T @ X_test
hidden_nodes_test = sigmoid(net_hidden_test)
net_output_test = weights_output.T @ hidden_nodes_test
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
