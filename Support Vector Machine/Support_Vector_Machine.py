import cvxopt
import numpy as np


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
        alphas (numpy.ndarray): An array storing all the Lagrange multipliers of the model.
        support_alphas (numpy.ndarray): An array storing the Lagrange multipliers for the support vectors.
        support_vectors (numpy.ndarray): An array storing the support vectors.
        support_ys (numpy.ndarray): An array storing the binary labels (-1, 1) of the support vectors.
        b (float): The intercept of the model used in prediction.
        w (numpy.ndarray): An array storing the weights of the model. Only applicable for linear kernel.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
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
        '''Build linear kernel function.'''

        return np.dot(x1, x2)

    def polynomial_kernel(self, x1, x2):
        '''Build polynomial kernel function.'''

        return (1 + np.dot(x1, x2)) ** self.degree

    def radial_basis_function_kernel(self, x1, x2):
        '''Build gaussian radial basis function.'''

        return np.exp(-self.gamma * (np.linalg.norm(x1-x2) ** 2))

    def calculate_intercept(self, K, IsSV):
        '''Calculate the model intercept (b).'''

        #! b = 1/N * Σ(y_i - Σα_j * y_j * K(x_i, x_j))
        self.b = np.mean(self.support_ys - np.sum(self.support_alphas *
                         self.support_ys * K[IsSV][:, IsSV], axis=1))

    def calculate_weights(self):
        '''Calculate the model weights (w).'''

        if self.kernel == self.linear_kernel:
            #! w = Σα_i * y_i * x_i
            self.w = np.sum(
                (self.support_alphas * self.support_ys).reshape(-1, 1) * self.support_vectors, axis=0)
        else:
            self.w = None

    def fit(self, X, y):
        """
        Fit the model using input matrix and corresponding labels.
        Note, the input data matrix should have the shape of (sample_number, feature_number).
        """

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

    def predict_class(self, X):
        '''Return the models predicted class for each of the given instances.'''

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            return np.sign(self.decision_function(X))
