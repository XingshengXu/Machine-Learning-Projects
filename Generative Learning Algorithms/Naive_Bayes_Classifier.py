import numpy as np


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier (NBC) and Multinomial Naive Bayes Classifier (MNBC) are 
    generative learning algorithms used for binary classification. These algorithms 
    calculate the conditional probability of each feature given the output class, 
    and use these probabilities to make predictions. Both algorithms assume that the 
    features are conditionally independent given the output class.

    NBC algorithm assumes binary features like a Bernoulli distribution.

    MNBC version assumes features represent counts or frequencies like a Multinomial distribution.

    Args:
        algorithm (str): Specify the version of Naive Bayes to use.

    Attributes:
        X (np.array): Standardized feature matrix (training data).
        y (np.array): Output vector (training labels).
        X_y0_prob (np.array): The conditional probabilities of each feature given y=0.
        X_y1_prob (np.array): The conditional probabilities of each feature given y=1.
        y0_prob (float): The prior probability for class 0.
        y1_prob (float): The prior probability for class 1.
        IsFitted (bool): Boolean flag to indicate if the model is trained.
    """

    def __init__(self, algorithm='NBC'):
        self.algorithm = algorithm
        self.IsFitted = False

    def fit(self, X, y):
        """Train the model with the given training set."""

        self.X = X
        self.y = y

        # Return all rows except its last column when y=0
        X_y0 = self.X[self.y == 0]

        # Return all rows except its last column when y=1
        X_y1 = self.X[self.y == 1]

        if self.algorithm == 'NBC':
            # Swap 0s with 1s and 1s with 0s in output set for summation of indicator functions
            swapped_y = np.where(self.y == 0, 1, 0)

            # ! φx1|y0 = ΣI(x(i)=1 & y(i)=0) / ΣI(y(i)=0)
            self.X_y0_prob = np.sum(X_y0, axis=0) / np.sum(swapped_y)

            # ! φx1|y1 = ΣI(x(i)=1 & y(i)=1) / ΣI(y(i)=1)
            self.X_y1_prob = np.sum(X_y1, axis=0) / np.sum(self.y)

        elif self.algorithm == 'MNBC':
            # ! φx|y0 = (Σx(i) where y(i)=0 + 1) / (Total count of words where y=0 + num_features)
            self.X_y0_prob = (np.sum(X_y0, axis=0) + 1) / (
                np.sum(X_y0) + len(X_y0[0]))

            # ! φx|y1 = (Σx(i) where y(i)=1 + 1) / (Total count of words where y=1 + num_features)
            self.X_y1_prob = (np.sum(X_y1, axis=0) + 1) / (
                np.sum(X_y1) + len(X_y1[0]))
        else:
            raise ValueError(
                f'{self.algorithm} is not a valid algorithm. Available algorithms: NBC, MNBC.')

        # ! φy1 = ΣI(y(i)=1) / n
        self.y1_prob = np.sum(self.y) / len(self.y)

        # ! φy0 = 1 - φy1
        self.y0_prob = 1 - self.y1_prob

        self.IsFitted = True

    def predict_class(self, X):
        """Predict the class for given input data."""

        if not self.IsFitted:
            raise ValueError(
                "Model is not fitted, call 'fit' with appropriate arguments before using model.")
        else:
            if self.algorithm == 'NBC':
                # Calculate the probabilities of input data using NBC.
                #! p(y=0|x) = p(y=0) * Π[p(x=1|y=0)^(x) * p(x=0|y=0)^(1-x)]
                p_y0_given_X = self.y0_prob * \
                    np.prod(np.power(self.X_y0_prob, X) *
                            np.power(1 - self.X_y0_prob, 1 - X), axis=1)

                #! p(y=1|x) = p(y=1) * Π[p(x=1|y=1)^(x) * p(x=0|y=1)^(1-x)]
                p_y1_given_X = self.y1_prob * \
                    np.prod(np.power(self.X_y1_prob, X) *
                            np.power(1 - self.X_y1_prob, 1 - X), axis=1)

            elif self.algorithm == 'MNBC':
                # Calculate the probabilities of input data using MNBC.
                #! p(y=0|x) = p(y=0) * Π[p(word|y=0)^(count of word in x)]
                p_y0_given_X = self.y0_prob * \
                    np.prod(np.power(self.X_y0_prob, X), axis=1)

                #! p(y=1|x) = p(y=1) * Π[p(word|y=1)^(count of word in x)]
                p_y1_given_X = self.y1_prob * \
                    np.prod(np.power(self.X_y1_prob, X), axis=1)

            # The predicted class is the one where p_y1_given_X > p_y0_given_X.
            pred_y = np.where(p_y1_given_X > p_y0_given_X, 1, 0)
            return pred_y
