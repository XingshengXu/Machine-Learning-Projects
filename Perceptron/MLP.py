"""
Multi Layer Perceptron Classifier for Handwritten Digit Recognition
"""

import idx2numpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

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
