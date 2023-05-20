"""
Logistic Regression using Newton's Method for classificating marketing target.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns

# Load Data
market_data = pd.read_csv(
    'Logistic_Regression/dataset/Social_Network_Ads.csv', header=0)

# Settings
theta = np.zeros(4)
data_size = len(market_data)
iteration = 0
cost = cost_diff = np.inf
cost_prev = 0
tolerance = 0.001
max_iterations = 1000
cost_memo = []


def h_func(theta, X):
    """Hypothesis Function"""
    return 1 / (1 + np.exp(-theta.T @ X))


def cost_func(theta, X, Y):
    """Cost Function"""
    h = h_func(theta, X)
    return -1 * np.mean(Y * np.log(h) + (1 - Y) * np.log(1 - h))


def gradient(theta, X, Y):
    """Gradient of the Cost Function"""
    h = h_func(theta, X)
    return X @ (h - Y)


def hessian(theta, X):
    """Hessian Matrix"""
    h = h_func(theta, X)
    D = np.diag(h * (1 - h))
    return X @ D @ X.T


# Training Set
training_set_X = market_data.iloc[:, 1:4]
training_set_Y = market_data.iloc[:, 4]

# Change 'Gender' Column as 'Male':1 and 'Female':0
training_set_X['Gender'] = training_set_X['Gender'].replace(
    {'Male': 1, 'Female': 0})

# Standardize the features
scaler = StandardScaler()
training_set_X = scaler.fit_transform(training_set_X)

# Add A Column of Ones to The Training Set As Intercept Term
X = np.hstack((np.ones((data_size, 1)), training_set_X)).T

# Implement Logistic Regression Training

# Loop through the entire dataset for each epoch
while cost_diff >= tolerance and iteration <= max_iterations:
    grad = gradient(theta, X, training_set_Y)
    H = hessian(theta, X)
    theta -= np.linalg.inv(H) @ grad  # !Newton-Raphson update
    cost = cost_func(theta, X, training_set_Y)
    cost_diff = np.abs(cost_prev - cost)
    cost_memo.append(cost)
    cost_prev = cost
    iteration += 1

print(f"Training finished after {iteration} iterations.")
print(f"Theta values:{theta}")

# Plot cost vs. iteration
plt.plot(range(1, iteration + 1), cost_memo)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# Convert probabilities into classes
y_pred = h_func(theta, X).round()

# Create confusion matrix
cm = confusion_matrix(training_set_Y, y_pred)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate ROC curve
fpr, tpr, _ = roc_curve(training_set_Y, h_func(theta, X))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(
    training_set_Y, h_func(theta, X))
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="lower right")
plt.show()

# Plot Predicted Probability Distribution
plt.hist(h_func(theta, X), bins=10, label='Predicted probabilities')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc="upper right")
plt.show()
