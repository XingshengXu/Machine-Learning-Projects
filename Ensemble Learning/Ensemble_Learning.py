import numpy as np
import path_setup
from Decisiontree import ClassificationTree, RegressionTree
from decision_tree_utils import *


class VotingClassifier:
    '''
    A VotingClassifier is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction based on either hard or 
    soft voting.

    Args:

    Attributes:

    '''

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
                preds = [classifier.predict_proba(
                    X) for classifier in self.classifiers]
                avg_preds = np.mean(preds, axis=0)
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
    equal to the mean predicted target value of all of the base estimators in 
    the ensemble.

    Args:

    Attributes:
    '''

    def __init__(self, regressors):
        self.regressors = regressors

    def fit(self, X, y):
        """Fit each of the base regressors to the training set."""

        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict_value(self, X):
        """Predict the output value of input data based on average value for ensemble model."""

        pred = np.mean([regressor.predict_value(X)
                       for regressor in self.regressors], axis=0)
        return pred
