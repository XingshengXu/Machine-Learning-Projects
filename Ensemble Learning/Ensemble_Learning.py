import numpy as np
import path_setup
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Decisiontree import ClassificationTree, RegressionTree
from Logistic_Regression import LogisticRegression


class VotingClassifier:
    """
    A Voting Classifier is an ensemble model that aggregates the predictions of 
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
            pred_y = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

            return pred_y


class VotingRegressor:
    '''
    A Voting Regressor is an ensemble model that aggregates the predictions of 
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
        """Predict the target value of input data based on average value for ensemble model."""

        # Reshape all predictions to 1D arrays
        preds = [np.reshape(regressor.predict_value(X), -1)
                 for regressor in self.regressors]

        # Compute and return the mean prediction
        pred_y = np.mean(preds, axis=0)
        return pred_y


class StackingClassifier:
    """
    A Stacking Classifier is an ensemble model in which the predictions of the
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

        # Train the first layer base classifiers
        for classifier in self.classifiers:
            classifier.fit(X_base, y_base)

        # Create training set for the meta-model
        train_set = np.column_stack(
            [classifier.predict_class(X_meta) for classifier in self.classifiers])

        # Train the meta-model
        self.final_classifier.fit(train_set, y_meta)

    def predict_class(self, X):
        """Predict the class of the input data based on the meta-model."""

        base_set = np.column_stack([classifier.predict_class(X)
                                   for classifier in self.classifiers])
        return self.final_classifier.predict_class(base_set)


class StackingRegressor:
    """
    A Stacking Regressor is an ensemble model in which the predictions of the 
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

        # Train the first layer base regressors
        for regressor in self.regressors:
            regressor.fit(X_base, y_base)

        # Create training set for the meta-model
        train_set = np.column_stack(
            [regressor.predict_value(X_meta) for regressor in self.regressors])

        # Train the meta-model
        self.final_regressor.fit(train_set, y_meta)

    def predict_value(self, X):
        """Predict the target value of input data based on the meta-model."""

        base_set = np.column_stack([regressor.predict_value(X)
                                    for regressor in self.regressors])
        return self.final_regressor.predict_value(base_set)


class Bagging:
    """
    An implementation of the Bagging ensemble method. This class serves as the 
    super class for the BaggingClassifier, BaggingRegressor, RandomForestClassifier, 
    and RandomForestRegressor classes.

    Args:
        base_model (object): The base model to be used in the ensemble.
        model_type (str): The type of model - 'classifier' or 'regressor'.
        model_number (int): The number of base models to be used in the ensemble.
        feature_proportion (float): The fraction of the total number of features to be 
                                     used to train each base model in the ensemble.

    Attributes:
        oob_score (float): The out-of-bag score of the ensemble. It is calculated as
                           the average OOB score of all base models.
        features (list): A list storing the indices of the features used to train each
                         of the base models in the ensemble.
        base_models (list): A list storing all of the trained base models in the ensemble.
        X (np.array): The input data used to train the ensemble.
        y (np.array): The target classes or values used to train the ensemble.
    """

    def __init__(self, base_model=None, model_type='classifier', model_number=10, feature_proportion=1.0):
        self.base_model = base_model
        self.base_models = []
        self.oob_score = 0
        self.features = []

        if model_type == 'classifier' or model_type == 'regressor':
            self.model_type = model_type
        else:
            raise AttributeError(
                "Invalid model type, 'classifier' and 'regressor' are available.")

        if model_number > 0:
            self.model_number = model_number
        else:
            raise AttributeError(
                "Invalid number of base models. 'model_number' argument not valid.")

        if feature_proportion == 'sqrt' or (feature_proportion > 0 and feature_proportion <= 1):
            self.feature_proportion = feature_proportion
        else:
            raise AttributeError(
                "Invalid proportion of features. 'feature_proportion' argument not valid.")

    def __get_estimator_params(self):
        """Extract the attributes of the base estimator to be used in the ensemble model."""

        dic = self.base_model.__dict__
        args = self.base_model.__init__.__code__.co_varnames
        params = {}

        for i in dic:
            if i in args:
                params[i] = dic[i]
        return params

    def __get_features(self, X):
        """Selects a random subset of features from the input data."""

        if self.feature_proportion == 'sqrt':
            feature_number = int(np.sqrt(X.shape[1]))
        else:
            feature_number = int(self.feature_proportion * X.shape[1])

        feature_idx = np.random.choice(
            X.shape[1], feature_number, replace=False)

        return feature_idx

    def __get_samples(self, X, y):
        """Generates in-bag samples and out-of-bag (OOB) samples."""

        sample_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        oob_idx = [i for i in range(X.shape[0]) if i not in sample_idx]

        return X[sample_idx], X[oob_idx], y[sample_idx], y[oob_idx]

    def fit(self, X, y):
        """Fit each of the base models in the ensemble model to the training set."""

        # Store the input data
        self.X = np.array(X)
        self.y = np.array(y)

        # Check if X is 1D and if so, reshape it into a 2D array
        if len(self.X.shape) == 1:
            self.X = self.X.reshape(-1, 1)

        # Get the parameters to be passed to the estimator's constructor.
        param_dict = self.__get_estimator_params()

        for _ in range(self.model_number):

            # Get random subset of features
            feature_idx = self.__get_features(self.X)

            # Generate in-bag samples (for training) and out-of-bag samples (for calculating OOB score)
            X_inbag, X_oob, y_inbag, y_oob = self.__get_samples(
                self.X[:, feature_idx], self.y)

            # Create a fresh clone of the base model
            model = self.base_model.__class__(**param_dict)

            # Train the cloned base model with the in-bag samples
            model.fit(X_inbag, y_inbag)

            # If there are any out-of-bag samples, calculate the OOB score
            if y_oob.size > 0:
                # For classifiers, calculate accuracy as OOB score
                if self.model_type == 'classifier':
                    pred_y = model.predict_class(X_oob)
                    score = np.sum(pred_y == y_oob) / len(y_oob)
                else:
                    # For regressors, calculate R-squared score as OOB score
                    pred_y = model.predict_value(X_oob)
                    score = r2_score(y_oob, pred_y)

                # Accumulate the OOB score
                self.oob_score += score

            # Store the selected features
            self.features.append(feature_idx)

            # Store the trained base model
            self.base_models.append(model)

        # Calculate the average OOB score across all base models
        self.oob_score /= self.model_number


class BaggingClassifier(Bagging):
    """
    A Bagging Classifier is an ensemble model that aggregates the predictions of
    multiple base models in order to make a final prediction. The base models
    are trained on random subsets (Bootstrapping) of the training data. The final 
    prediction is calculated making use of hard voting of all of the base models 
    in the ensemble model. 

    Args:
        base_model (object): The base model to be used in the ensemble.
        model_type (str): The type of model - 'classifier' or 'regressor'.
        model_number (int): The number of base models to be used in the ensemble.
        feature_proportion (float): The fraction of the total number of features to be 
                                     used to train each base model in the ensemble.

    Attributes:
        oob_score (float): The out-of-bag score of the ensemble. It is calculated as
                           the average OOB score of all base models.
        features (list): A list storing the indices of the features used to train each
                         of the base models in the ensemble.
        base_models (list): A list storing all of the trained base models in the ensemble.
        X (np.array): The input data used to train the ensemble.
        y (np.array): The target classes or values used to train the ensemble.
    """

    def predict_class(self, X):
        """Predict the class of the samples based on majority frequency for rnsemble model."""

        X = np.array(X)

        # If X is 1D, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Get predictions from each model
        preds = np.array([model.predict_class(X[:, self.features[i]])
                         for i, model in enumerate(self.base_models)])

        # Transpose so each row corresponds to a sample and each column to a base model's prediction
        preds = np.transpose(preds).astype(int)

        # Find mode for each sample
        pred_y = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

        return pred_y


class BaggingRegressor(Bagging):
    """
    A Bagging Regressor is an ensemble model that aggregates the predictions of 
    multiple base models and makes a final prediction. The base models are trained 
    on random subsets (Bootstrapping) of the training data.The final prediction is 
    equal to the mean predicted target value of all of the base models in the 
    ensemble model. 

    Args:
        base_model (object): The base model to be used in the ensemble.
        model_type (str): The type of model - 'classifier' or 'regressor'.
        model_number (int): The number of base models to be used in the ensemble.
        feature_proportion (float): The fraction of the total number of features to be 
                                     used to train each base model in the ensemble.

    Attributes:
        oob_score (float): The out-of-bag score of the ensemble. It is calculated as
                           the average OOB score of all base models.
        features (list): A list storing the indices of the features used to train each
                         of the base models in the ensemble.
        base_models (list): A list storing all of the trained base models in the ensemble.
        X (np.array): The input data used to train the ensemble.
        y (np.array): The target classes or values used to train the ensemble.
    """

    def predict_value(self, X):
        """Predict the target value of the samples based on average value for ensemble model."""

        X = np.array(X)

        # If X is 1D, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Get predictions from each model
        preds = [np.reshape(model.predict_value(X[:, self.features[i]]), -1)
                 for i, model in enumerate(self.base_models)]

        # Compute and return the mean prediction
        pred_y = np.mean(preds, axis=0)

        return pred_y


class RandomForestClassifier(Bagging):
    """
    A Random Forest Classifier is a type of ensemble model that utilizes multiple 
    decision trees by constructing these multiple trees at the training phase and 
    outputting the class that is the mode of the classes produced by individual tree 
    at prediction. It introduces additional randomness of selecting features when 
    growing trees to ensure diversity among the trees and mitigate overfitting. The 
    final prediction is calculated making use of hard voting of all of the decision 
    trees in the ensemble model. 

    Args:
        base_model (object): The base model to be used in the ensemble.
        model_type (str): The type of model - 'classifier' or 'regressor'.
        model_number (int): The number of base models to be used in the ensemble.
        feature_proportion (float): The fraction of the total number of features to be 
                                     used to train each base model in the ensemble.

    Attributes:
        oob_score (float): The out-of-bag score of the ensemble. It is calculated as
                           the average OOB score of all base models.
        features (list): A list storing the indices of the features used to train each
                         of the base models in the ensemble.
        base_models (list): A list storing all of the trained base models in the ensemble.
        X (np.array): The input data used to train the ensemble.
        y (np.array): The target classes or values used to train the ensemble.
    """

    def __init__(self, model_number=10, feature_proportion='sqrt'):
        super().__init__(base_model=ClassificationTree(),
                         model_type='classifier',
                         model_number=model_number,
                         feature_proportion=feature_proportion)

    def predict_class(self, X):
        """Predict the class of the samples based on majority frequency for ensemble model."""

        X = np.array(X)

        # If X is 1D, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Get predictions from each model
        preds = np.array([model.predict_class(X[:, self.features[i]])
                         for i, model in enumerate(self.base_models)])

        # Transpose so each row corresponds to a sample and each column to a base model's prediction
        preds = np.transpose(preds).astype(int)

        # Find mode for each sample
        pred_y = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=1, arr=preds)

        return pred_y


class RandomForestRegressor(Bagging):
    """
    A Random Forest Regressor is a type of ensemble model that utilizes multiple 
    decision trees by constructing these multiple trees at the training phase and 
    outputting the class that is the mode of the classes produced by individual tree 
    at prediction. It introduces additional randomness of selecting features when 
    growing trees to ensure diversity among the trees and mitigate overfitting. The 
    final prediction made by the ensemble is equal to the mean predicted target value 
    of all of the decision trees in the ensemble model.

    Args:
        base_model (object): The base model to be used in the ensemble.
        model_type (str): The type of model - 'classifier' or 'regressor'.
        model_number (int): The number of base models to be used in the ensemble.
        feature_proportion (float): The fraction of the total number of features to be 
                                     used to train each base model in the ensemble.

    Attributes:
        oob_score (float): The out-of-bag score of the ensemble. It is calculated as
                           the average OOB score of all base models.
        features (list): A list storing the indices of the features used to train each
                         of the base models in the ensemble.
        base_models (list): A list storing all of the trained base models in the ensemble.
        X (np.array): The input data used to train the ensemble.
        y (np.array): The target classes or values used to train the ensemble.
    """

    def __init__(self, model_number=10, feature_proportion='sqrt'):
        super().__init__(base_model=RegressionTree(),
                         model_type='regressor',
                         model_number=model_number,
                         feature_proportion=feature_proportion)

    def predict_value(self, X):
        """Predict the target value of the samples based on average value for ensemble model."""

        X = np.array(X)

        # If X is 1D, reshape it to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # Get predictions from each model
        preds = [np.reshape(model.predict_value(X[:, self.features[i]]), -1)
                 for i, model in enumerate(self.base_models)]

        # Compute and return the mean prediction
        pred_y = np.mean(preds, axis=0)

        return pred_y


class GradientBoostingRegressor:
    """
    A GradientBoostingRegressor is an ensemble model that sequentially trains base models.
    Each successive model is trained on the pseudo-residuals of the combined previous base models. 
    This iterative correction of errors helps in reducing bias of the model predictions.

    Args:
        model_number (int): The number of base models (trees) to be used in the ensemble.
        learning_rate (float): A factor to shrink the contribution of each base model.
        max_depth (int): The maximum depth of each base model (regression tree).

    Attributes:
        max_depth (int): Maximum depth for the base regression trees.
        models (list): List containing the base models (regression trees) and the initial prediction.
    """

    def __init__(self, model_number=100, learning_rate=0.1, max_depth=4):
        self.model_number = model_number
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def __calculate_residuals(self, X, y):
        """Private function to calculate the pseudo-residuals based on current ensemble prediction."""

        residuals = y - self.predict_value(X)
        return residuals

    def fit(self, X, y):
        """Fit each of the base models in the ensemble model to the training set."""

        # Starting prediction is the mean of target variable
        initial_pred = np.mean(y)
        self.models.append(initial_pred)

        # Iteratively fit trees on residuals
        for _ in range(self.model_number - 1):
            residuals = self.__calculate_residuals(X, y)
            tree = RegressionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.models.append(tree)

    def predict_value(self, X):
        """Predict the target value for the given samples by contribution of the entire ensemble model."""

        # Start with the initial prediction for all samples
        pred_y = np.full(X.shape[0], self.models[0])

        # Add contribution from each tree
        for model in self.models[1:]:
            pred_y += self.learning_rate * model.predict_value(X)
        return pred_y
