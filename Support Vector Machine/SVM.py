import cvxopt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix


def generate_test_data(n_samples=100, n_features=2, std=0.5):
    '''Generate a test dataset with n-dimensional instances.'''
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2,
                      random_state=0, cluster_std=std)
    y = convert_labels(y)
    return X, y


def convert_labels(Y):
    """Convert labels from {0,1} to {-1,1} and change the data type to float."""
    Y[Y == 0] = -1
    return Y.astype(np.float64)


def create_contour_plot(svm, X, y, kernel='linear', filled=False):
    '''Generate a contour plot for the given model using the dataset X with class labels y.'''
    x_min, x_max = np.min(X[:, 0]) - 0.5, np.max(X[:, 0]) + 0.5
    y_min, y_max = np.min(X[:, 1]) - 0.5, np.max(X[:, 1]) + 0.5

    X1, X2 = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    X_grid = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.decision_function(X_grid).reshape(X1.shape)

    plt.figure(figsize=(10, 7))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if filled:
        cp = plt.contourf(X1, X2, Z, cmap='coolwarm')
        colorbar = plt.colorbar(cp)
        colorbar.set_label('Prediction Confidence')
    else:
        cp = plt.contour(
            X1, X2, Z, [-1, 0, 1], colors=['black', 'fuchsia', 'black'], linewidths=2)
        plt.clabel(cp, inline=True, fontsize=10)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    plt.scatter(svm.support_vectors[:, 0], svm.support_vectors[:, 1],
                facecolors='none', edgecolors='green', marker='o', s=100)
    plt.xlabel('Feature One')
    plt.ylabel('Feature Two')
    plt.title(
        f'Support Vector Machine Classifier\nKernel: {kernel} C= {svm.C}')
    plt.show()


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


class SupportVectorMachine():
    '''
    A Support Vector Machine (SVM) is a classifier that finds the hyperplane 
    which best separates the classes in the training data. The model predicts 
    the class of an instance based on its position relative to the hyperplane 
    (the decision boundary).

    Args:
        kernel (str, optional): The kernel function used by the SVM. 
            Available options are: 
            'linear' - Linear kernel (default)
            'polynomial' - Polynomial kernel
            'rbf' - Gaussian Radial Basis Function (RBF) kernel
        C (float, optional): The regularization parameter which controls the trade-off between 
            maximizing the margin and minimizing the classification error. A larger value of `C` 
            gives the optimization a higher penalty for misclassified data points, leading to 
            a model with higher variance but lower bias. Default is None (no regularization).
        gamma (float, optional): A hyperparameter for the RBF kernel. A larger `gamma` 
            value makes the model more complex, potentially leading to overfitting. Default is 1.
            Ignored by kernels other than the RBF kernel.
        degree (int, optional): The degree of the polynomial for the 'polynomial' kernel.
            Default is 3. Ignored by kernels other than the 'polynomial' kernel.

    Attributes:
        kernel (str): The kernel function used by the SVM.
        C (float): The value of the `C` hyperparameter of the SVM.
        gamma (float): The value of the `gamma` hyperparameter of the SVM.
        degree (int): The value of the `degree` hyperparameter of the SVM.
        alphas (numpy.ndarray): An array storing all the Lagrange multipliers of the model.
        support_alphas (numpy.ndarray): An array storing the Lagrange multipliers for the support vectors.
        support_vectors (numpy.ndarray): An array storing the support vectors.
        support_ys (numpy.ndarray): An array storing the binary labels (-1, 1) of the support vectors.
        b (float): The intercept of the model used in prediction.
        w (numpy.ndarray): An array storing the weights of the model. Only applicable for linear kernel.
    '''

    def __init__(self, kernel='linear', C=None, gamma=1, degree=3):
        self.gamma = gamma
        self.degree = degree
        self.IsFitted = False

        # Only allow valid kernel functions
        kernels = {'linear': self.linear_kernel,
                   'polynomial': self.polynomial_kernel,
                   'rbf': self.radial_basis_function_kernel
                   }
        try:
            self.kernel = kernels[kernel]
        except KeyError:
            raise ValueError(
                f'{kernel} is not a valid kernel function. Available kernels: linear, polynomial and rbf.')

        # Polynomial kernel SVM with degree 1 is equivalent to a linear kernel SVM.
        if degree == 1:
            self.kernel = kernels['linear']

        # Hyperparameter C should be a positive number
        if C is not None and C > 0:
            self.C = np.float64(C)
        else:
            self.C = None

    def linear_kernel(self, x1, x2):
        '''Linear kernel function.'''
        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2):
        '''Polynomial kernel function.'''
        return (1 + np.dot(x1, x2)) ** self.degree

    def radial_basis_function_kernel(self, x1, x2):
        '''Gaussian Radial Basis Function.'''
        return np.exp(-self.gamma * (np.linalg.norm(x1-x2) ** 2))

    def calculate_intercept(self, K, IsSV):
        '''Calculate the model intercept (b)'''
        #! b = 1/N * Σ(y_i - Σα_j * y_j * K(x_i, x_j))
        self.b = np.mean(self.support_ys - np.sum(self.support_alphas *
                         self.support_ys * K[IsSV][:, IsSV], axis=1))

    def calculate_weights(self):
        '''Calculate the model weights (w)'''
        if self.kernel == self.linear_kernel:
            #! w = Σα_i * y_i * x_i
            self.w = np.sum(
                (self.support_alphas * self.support_ys).reshape(-1, 1) * self.support_vectors, axis=0)
        else:
            self.w = None

    def fit(self, X, y):
        n = X.shape[0]

        # Apply the kernel function resulting in a Gram matrix K
        K = np.array([[self.kernel(X[i], X[j]) for j in range(n)]
                     for i in range(n)])

        # Setup necessary matrices to be used in the calculation of the Lagrangian multipliers.
        #! minimize (1/2)*x'*P*x + q'*x, s.t., G*x <= h, A*x = b
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((n, 1)))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(np.zeros((1, 1)))

        if self.C is None:
            h = cvxopt.matrix(np.zeros((n, 1)))
            G = cvxopt.matrix(-np.eye(n))
        else:
            G_lower = -np.eye(n)
            G_upper = np.eye(n)
            h_lower = np.zeros((n, 1))
            h_upper = np.full((n, 1), self.C)
            G = cvxopt.matrix(np.concatenate((G_lower, G_upper), axis=0))
            h = cvxopt.matrix(np.concatenate((h_lower, h_upper), axis=0))

        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract the Lagrangian multipliers.
        self.alphas = (np.array(solution['x'])).flatten()

        # Determine the support vectors, their associated Lagrangian multipliers and labels.
        IsSV = self.alphas > 1e-4
        self.support_alphas_indices = np.arange(n)[IsSV]
        self.support_alphas = self.alphas[IsSV]
        self.support_vectors = X[IsSV]
        self.support_ys = y[IsSV]

        # Calculate the model intercept.
        self.calculate_intercept(K, IsSV)

        # Calculate the model weights if applicable.
        self.calculate_weights()

        # The model has been fitted
        self.IsFitted = True

    def decision_function(self, X):
        '''Calculate and return the real valued prediction of the model.'''
        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")

        if self.w is not None:
            #! f(x) = w^T * x + b
            return X @ self.w + self.b
        else:
            #! f(x) = Σ(α_i * y_i * K(x_i, x_j)) + b
            kernel_vals = np.array([self.kernel(
                x, sv) for sv in self.support_vectors for x in X]).reshape(-1, len(X)).T
            preds = np.sum((self.support_alphas * self.support_ys)
                           * kernel_vals, axis=1)
            return preds + self.b

    def predict(self, X):
        '''Return the models predicted class for each of the given instances.'''
        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            return np.sign(self.decision_function(X))
