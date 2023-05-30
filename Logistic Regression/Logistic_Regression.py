"""
Logistic Regression for Classificating Marketing Target
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from matplotlib.colors import ListedColormap

# Load Data
market_data = pd.read_csv(
    'Logistic Regression/dataset/Social_Network_Ads.csv', header=0)

# Settings
theta = np.zeros(3)
iteration = 0
cost_prev = cost_diff = np.inf
cost = 0
learning_rate = 0.01
tolerance = 0.001
max_iterations = 1000
cost_memo = []


def h_func(theta, X):
    """Hypothesis Function"""
    return 1 / (1 + np.exp(-theta.T @ X))


def gradient(theta, X, Y):
    """Gradient of the Cost Function"""
    h = h_func(theta, X)
    return X @ (Y - h)


def cost_func(theta, X, Y):
    """Cost Function"""
    h = h_func(theta, X)
    return -1 * np.mean(Y * np.log(h) + (1 - Y) * np.log(1 - h))


# Remove the First Column
market_data = market_data.drop(market_data.columns[0], axis=1)

# Training Set
training_set_X = market_data.iloc[0:300, 0:-1]
training_set_Y = market_data.iloc[0:300:, -1]

# Test Set
test_set_X = market_data.iloc[300:, 0:-1]
test_set_Y = market_data.iloc[300:, -1]

# Standardize the Features
scaler = StandardScaler()
training_set_X = scaler.fit_transform(training_set_X)
test_set_X = scaler.transform(test_set_X)

# Add A Column of Ones to The Training Set As Intercept Term
training_set_X = np.hstack((np.ones((300, 1)), training_set_X)).T
test_set_X = np.hstack((np.ones((100, 1)), test_set_X)).T

# Impletement Logistic Regression Training

# Loop Through the Entire Dataset for Each Epoch
while cost_diff >= tolerance and iteration <= max_iterations:
    grad = gradient(theta, training_set_X, training_set_Y)
    theta += learning_rate * grad  # !Batch Gradient Descent update
    cost = cost_func(theta, training_set_X, training_set_Y)
    cost_diff = np.abs(cost_prev - cost)
    cost_memo.append(cost)
    cost_prev = cost
    iteration += 1

print(f"Training finished after {iteration} iterations.")
print(f"Theta values:{theta}")

# Create Grid
x_values = np.linspace(-3, 3, 500)
y_values = (-theta[0] - (theta[1]*x_values)) / theta[2]

# Create a Colormap
cmap = ListedColormap(['red', 'blue'])

# Plotting the Data
plt.scatter(training_set_X[1, :], training_set_X[2, :],
            c=training_set_Y, cmap=cmap, edgecolors='k')
plt.plot(x_values, y_values, label='Decision Boundary', c='green')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Training Data with Decision Boundary')
plt.legend()
plt.show()

# Plot Cost vs. Iteration
plt.plot(range(1, iteration + 1), cost_memo)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.show()

# Convert Probabilities Into Classes
y_pred = h_func(theta, test_set_X).round()

# Print Classification Report
print(classification_report(test_set_Y, y_pred, zero_division=0))

# Create Confusion Matrix
cm = confusion_matrix(test_set_Y, y_pred)

# Plot Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Calculate ROC Curve
fpr, tpr, _ = roc_curve(test_set_Y, h_func(theta, test_set_X))
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
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
    test_set_Y, h_func(theta, test_set_X))
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
plt.hist(h_func(theta, test_set_X), bins=10,
         label='Predicted probabilities')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.legend(loc="upper right")
plt.show()
