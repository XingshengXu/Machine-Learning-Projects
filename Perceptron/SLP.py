"""
Single Layer Perceptron Classifier for Handwritten Digit Recognition
"""

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def grayscale_to_binary(image, threshold):
    """Greyscale to Binary Image Function"""
    filter_image = image >= threshold
    return filter_image


# Load Training and Testing Data Sets
train_set = idx2numpy.convert_from_file(
    './Perceptron/dataset/train-images.idx3-ubyte')
train_label = idx2numpy.convert_from_file(
    './Perceptron/dataset/train-labels.idx1-ubyte')
test_set = idx2numpy.convert_from_file(
    './Perceptron/dataset/t10k-images.idx3-ubyte')
test_label = idx2numpy.convert_from_file(
    './Perceptron/dataset/t10k-labels.idx1-ubyte')

# Parameter Settings
threshold = 128  # grayscale to binary thresholding
sigma = 0  # output thresholding
image_size = 28
learning_rate = 0.1
max_iterations = 100
train_set_size = len(train_set)
test_set_size = len(test_set)
theta = np.random.rand(image_size ** 2, 10)
X_train = np.zeros((image_size ** 2, train_set_size))
X_test = np.zeros((image_size ** 2, test_set_size))

# Initializations
cost = 0
cost_memo = []
iteration = 0

# Transfer Images from Grayscale to Binary
train_images = grayscale_to_binary(train_set, threshold)
test_images = grayscale_to_binary(test_set, threshold)

# Reshape Image Size From 2D to 1D
X_train = train_images.reshape(train_set_size, -1).T
X_test = test_images.reshape(test_set_size, -1).T

# Convert Labels to Expected Outputs
Y_train = np.eye(10)[train_label].T


# Single Layer Perceptron Training
while iteration <= max_iterations:
    h_func = theta.T @ X_train
    Y_pred = h_func >= sigma
    error = Y_train - Y_pred
    grad = X_train @ error.T
    theta += learning_rate * grad
    cost = np.sum(error ** 2) / (2 * train_set_size)
    cost_memo.append(cost)
    iteration += 1
    print(f'{iteration} iterations')

print(f"Training finished after {iteration} iterations.")
print(f"Theta values:{theta}")

# Plot Cost vs. Iteration
plt.plot(range(1, iteration + 1), cost_memo)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# Single Layer Perceptron Testing
h_func_test = theta.T @ X_test
Y_pred_test = h_func_test >= sigma

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
