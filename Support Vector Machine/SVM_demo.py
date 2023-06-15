import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from sklearn.datasets import make_blobs
from SVM import SupportVectorMachine


def generate_test_data(n_samples=100, n_features=2, std=0.5):
    '''Generate a test dataset with n-dimensional instances.'''
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2,
                      random_state=0, cluster_std=std)
    y[y == 0] = -1
    y = np.ones(len(X)) * y
    return X, y


def create_contour_plot(svm, X, y, kernel='linear', filled=False):
    '''Generate a contour plot for the given model using the dataset X with class labels y.'''
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))

    Z = svm.decision_function(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.figure(figsize=(10, 7))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if filled:
        cp = plt.contourf(X1, X2, Z, cmap='coolwarm')
        plt.colorbar(cp)
    else:
        cp = plt.contour(
            X1, X2, Z, [-1, 0.0, 1], colors=['black', 'fuchsia', 'black'], linewidths=2)
        plt.clabel(cp, inline=True, fontsize=10)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                facecolors='none', edgecolors='green', marker='o', s=100)
    plt.xlabel('Feature One')
    plt.ylabel('Feature Two')
    plt.title(f'Support Vector Machine Classifier\nKernel: {kernel}')
    plt.show()


'''
Display a plot that illustrates the margin and decision boundary of a SVM
with a linear kernel on linearly separable data.
'''

X, y = generate_test_data(std=0.2)
svm = SupportVectorMachine(C=0)
svm.fit(X, y)
create_contour_plot(svm, X, y)
