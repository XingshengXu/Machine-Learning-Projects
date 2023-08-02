import numpy as np
import path_setup
from sklearn.model_selection import train_test_split
from Decisiontree import RegressionTree
from Logistic_Regression import LogisticRegression


class VotingClassifier:
    """
    A VotingClassifier is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction based on either hard or 
    soft voting.

    Args:
        classifiers (list): A list of classifiers to be used in the ensemble.
        voting (str, optional): The voting method to be used, 'hard' or 'soft'.
    """

    def __init__(self, classifiers, voting='hard'):
        self.classifiers = classifiers

        if voting == 'hard' or voting == 'soft':
            self.voting = voting
        else:
            raise AttributeError(
                "Invalid 'voting' argument, 'hard' and 'soft' voting available.")

    def fit(self, X, y):
        """Fit each of the base classifiers to the training set."""

        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_class(self, X):
        """Predict the class of the input data based on either soft voting or hard voting."""

        if self.voting == 'soft':
            try:
                # Generate the class probabilities for each classifier in the ensemble
                preds = [classifier.predict_proba(
                    X) for classifier in self.classifiers]

                # Calculate the average predicted probabilities across all classifiers
                avg_preds = np.mean(preds, axis=0)

                # Find the class with the highest average probability
                return np.argmax(avg_preds, axis=1)
            except AttributeError:
                print(
                    "The list of specified estimators does not support soft voting, set voting to 'hard' to resolve.")
        else:
            preds = np.array([classifier.predict_class(X)
                             for classifier in self.classifiers])

            # Transpose to make each row represent one sample and each column represent predictions of one classifier
            preds = np.transpose(preds).astype(int)

            # Get the mode for each sample
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=preds)


class VotingRegressor:
    '''
    A VotingRegressor is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction. The final prediction is 
    equal to the mean predicted target value of all base models in the ensemble.

    Args:
        regressors (list): A list of regressors to be used in the ensemble model.
    '''

    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        """Fit each of the base regressors to the training set."""

        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict_value(self, X):
        """Predict the output value of input data based on average value for ensemble model."""

        # Reshape all predictions to 1D arrays
        preds = [np.reshape(regressor.predict_value(X), -1)
                 for regressor in self.regressors]

        # Compute and return the mean prediction
        pred = np.mean(preds, axis=0)
        return pred


class StackingClassifier:
    """
    A StackingClassifier is an ensemble model in which the predictions of the
    base classifiers are aggregated through the use of the Meta-Classifier.

    Args:
        classifiers (list): A list of base classifiers that will be used in the ensemble model.
        final_classifier (object, optional): The meta-classifier to be used in the ensemble model.
    """

    def __init__(self, classifiers, final_classifier=None):
        self.classifiers = classifiers
        if final_classifier is not None:
            self.final_classifier = final_classifier
        else:
            self.final_classifier = LogisticRegression()

    def fit(self, X, y):
        """Fit each of the base classifiers to the training set."""

        X_base, X_meta, y_base, y_meta = train_test_split(
            X, y, test_size=0.3, random_state=0)

        # Train the first layer base classifiers.
        for classifier in self.classifiers:
            classifier.fit(X_base, y_base)

        # Create training set for the meta-model.
        train_set = np.column_stack(
            [classifier.predict_class(X_meta) for classifier in self.classifiers])

        # Train the meta-model.
        self.final_classifier.fit(train_set, y_meta)

    def predict_class(self, X):
        """Predict the class of the input data based on the meta-model."""

        base_set = np.column_stack([classifier.predict_class(X)
                                   for classifier in self.classifiers])
        return self.final_classifier.predict_class(base_set)


class StackingRegressor:
    """
    A StackingRegressor is an ensemble model in which the predictions of the 
    regressors are aggregated through the use of the Meta-Regressor.

    Args:
        regressors (list): A list of base regressors that will be used in the ensemble model.
        final_regressor (object, optional): The meta-regressor to be used in the ensemble model.
    """

    def __init__(self, regressors, final_regressor=None):
        self.regressors = regressors

        if final_regressor is not None:
            self.final_regressor = final_regressor
        else:
            self.final_regressor = RegressionTree()

    def fit(self, X, y):
        """Fit each of the base regressors to the training set."""

        X_base, X_meta, y_base, y_meta = train_test_split(
            X, y, test_size=0.3, random_state=0)

        # Train the first layer base regressors.
        for regressor in self.regressors:
            regressor.fit(X_base, y_base)

        # Create training set for the meta-model.
        train_set = np.column_stack(
            [regressor.predict_value(X_meta) for regressor in self.regressors])

        # Train the meta-model.
        self.final_regressor.fit(train_set, y_meta)

    def predict_value(self, X):
        """Predict the output value of input data based on the meta-model."""

        base_set = np.column_stack([regressor.predict_value(X)
                                    for regressor in self.regressors])
        return self.final_regressor.predict_value(base_set)
