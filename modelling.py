"""
F. P. Chmiel IT innovation centre 2020
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def fit_model(X, y, model=XGBClassifier, params={}, uid=None,
              cv_splitter=GroupKFold, nsplits=5):
    """
    Fits a model using given params and CV method and returns the fitted 
    classifier. If multiple-fold cross-validation is used each individual 
    classifier is returned within an iterable
    
    Parameters:
    -----------
    X : np.array ,
        Training data shape (n,m) where n is the number of instances and m the number of 
        features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    model : sklearn.base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
    
    params : dict,
        Parameter dictionary to be passed to the model.
    
    uid : np.array (default=None),
        User ids for each training instance, shape (n,). This is passed to the 
        cv_splitter so (for example) patient-wise validation folds can be 
        created.
    
    cv_splitter : cross-validation generator, 
        The cross-validation splitter (consitent with sklearn API)
        
    nsplits : int,
        The number of validation folds to create. Passed to cv_splitter.
        
    Returns:
    --------
    models, sklearn.base.BaseEstimater 
        Returns the fitted model(s)instance.
    """
    
    splitter = cv_splitter(n_splits=nsplits)

    models = []
    for train_index, test_index in splitter.split(X, y, uid):
        X_train, _ = X.values[train_index], X.values[test_index]
        y_train, _ = y.values[train_index], y.values[test_index]
        clf = model(**params)
        clf.fit(X_train, y_train)
        models.append(clf)
    return models

def tune_model(X, y, param_grid, uid=None, model=XGBClassifier, 
               cv_splitter=GroupKFold, n_splits=5, metric='average_precision'):
    """Tunes hyperparameters of a model by performing grid_search.
    
    Parameters:
    -----------
    X : np.array, 
        Training data shape (n,m) where n is the number of instances and m the
        number of features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    param_grid : dict, 
        Parameter dictionary to be passed to the model. Each value should be a 
        list of values of the specified parameter to search.
        
    uid : np.array (default=None),
        User ids for each training instance, shape (n,). This is passed to the 
        cv_splitter so (for example) patient-wise validation folds can be 
        created.
    
    model : sklearn.Base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
        
    cv_splitter : cross-validation generator (default=GroupKFold),
        The cross-validation splitter (consitent with sklearn API).
    
    nsplits : int,
        The number of validation folds to create. Passed to cv_splitter.
        
    Returns:
    --------
    clf : GridSearchCV,
        A fitted instance of a GridSearchCV class.
        
    """
    # instantiate model
    model = XGBClassifier()
    # get training / validation fold idxs
    cv_splits = None
    if cv_splitter is not None:
        cv_splitter = cv_splitter(n_splits=n_splits)
        cv_splits = []
        for train_index, test_index in cv_splitter.split(X, y, uid):
            cv_splits.append((train_index, test_index))
    hyperparameter = GridSearchCV(estimator=model, param_grid=param_grid, 
                                  scoring=metric, cv=cv_splits)
    hyperparameter.fit(X, y)
    
    return hyperparameter

def select_n_best_features(X, y, params={}, uid=None, N=10, 
                           model=XGBClassifier, cv_splitter=GroupKFold,
                           n_splits=5, metric='average_precision'):
    """Tunes hyperparameters of a model by performing grid_search.
    
    Parameters:
    -----------
    X : np.array, 
        Training data shape (n,m) where n is the number of instances and m the 
        number of features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    params : dict, 
        Parameter dictionary to be passed to the model.
        
    uid : np.array (default=None),
        User ids for each training instance, shape (n,). This is passed to the 
        cv_splitter so (for example) patient-wise validation folds can be 
        created.

    N : in (default=10)
        Number of features to select.

    model : sklearn.Base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
        
    cv_splitter : cross-validation generator (default=GroupKFold),
        The cross-validation splitter (consitent with sklearn API).
    
    nsplits : int,
        The number of validation folds to create. Passed to cv_splitter.
        
    metric : str (default='average_precision'),
        The metric to use in the feed-forward selection criteria.

    Returns:
    --------
    sfs : SFS,
        A fitted instance of a SequentialFeatureSelector class.
        
    """
    # get idxs for train/validation folds
    splitter = cv_splitter(n_splits=5)
    cv_idxs = []
    for train_index, test_index in splitter.split(X, y, uid):
        cv_idxs.append((train_index, test_index))
    
    sfs = SFS(model(**params), 
               k_features=N, 
               forward=True, 
               floating=False, 
               verbose=2,
               scoring=metric,
               cv=cv_idxs,
               n_jobs=2,
               fixed_features=(0,1))
    
    sfs = sfs.fit(X.values, y.values)
    
    print('Best features: {}'.format(X.columns[np.array(sfs.k_feature_idx_)]))
    
    return sfs

def predict_from_model(model, X):
    """
    Makes predictions using a pre-fitted model. If the provided model is an array-like object
    (i.e., a list of fitted models)
    
    Parameters:
    -----------
    model : {sklearn.base.BaseEstimater, array-like},
        A pre-fit model, compatiable with the sklearn API. If the model is a
        list of models the average prediction of each model is returned.
    Returns:
    --------
    preds : np.array,
        The predictions, length (n,) array.
    """
    try:
        preds = model.predict_proba(X)
    except AttributeError:
        # take average prediction of each model
        preds = np.zeros(len(X))
        num_models = len(model)
        for clf in model:
            preds += clf.predict_proba(X)[:,1] / num_models        
    return preds

class Heuristic(BaseEstimator, ClassifierMixin):

    def __init__(self, p1=0.019, p2=0.0757, p3=0.676, symptom_col=0):
        """
        A heuristic model for predicting risk of exacerbation.
        
        The model only uses symptom scores and classifiers a symptom score 
        of 1 as low-risk, 2 medium risk, 3  high-risk and 4 very-high risk

        Parameters:
        ----------
        params : NoneType
            Ignored. Included for compatabilty reasons.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.symptom_col = symptom_col
        # could make class method constructor to initate model using raw data

    def fit(self, X, y):
        """
        Ignored. This classifier is a heuristic model, this method is included 
        to be consistent with the sklearn API.
        """
        pass

    def predict(self, X):
        """
        Predicts the class of each instance in X using the BML heuristic model.

        Parameters:
        -----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The training data, the first column must be the symptom score.
        
        symptom_col: int,
            The integer index of the column containing self-reported symptom 
            scores.

        Returns:
        --------
        predictions : np.array
            The predictions with three possible values [0, 0.5, 1] corresponding
            to low, medium and high risk respectively.
        """
        symptom_scores = X[:, self.symptom_col]
        symptom_scores[symptom_scores==1] = self.p1 # low risk
        symptom_scores[symptom_scores==2] = self.p2 # medium risk
        symptom_scores[symptom_scores>2] = self.p3 # high risk 
        return np.vstack([1-symptom_scores, symptom_scores]).T

    def predict_proba(self, X, symptom_col=2):
        """
        Predicts the class of each instance in X using the BML heuristic model.

        This is exactly the same as calling the predict method.

        Parameters:
        -----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The training data, the first column must be the symptom score.

        Returns:
        --------
        predictions : np.array
            The predictions with three possible values [0, 0.5, 1] corresponding
            to low, medium and high risk respectively.
        """
        return self.predict(X)

class Ensemble(BaseEstimator, ClassifierMixin):

    def __init__(self, models, fitted=True, aggregation='mean'):
        """
        Takes a list of fitted sklear models and ensembles them by mean
        averaging.

        Parameters:
        -----------
        models : list
            A list fitted sklearn classifiers. If false the .fit method of
            this class will fit the models.

        fitted: bool
            If the models have been fitted.

        aggregation : str (default='mean')
            Method for aggregation of the model. Currently only mean is
            implemented.
        """
        self.models = models
        self.fitted = fitted

    def fit(self, X, y):
        """
        Ignored. This classifier takes pre-fitted models and so 

        Parameters:
        -----------

        X 
        """
        X, y = check_X_y(X, y)
        
        if not self.fitted:
            # fit the models
            for model in self.models:
                model.fit(X, y)
        else:
            pass

    def predict(self, X):
        """
        Predicts the class of each instance in X using the ensemble of 
        models.

        Parameters:
        -----------

        X : {array-like, sparse matrix}, shape [n_samples, n_features]

        y : {array-like, sparse matrix}, shape [n_samples, 1]

        Returns

        """
        X = check_array(X)
        # check each model is fitted
        for model in self.models:
            check_is_fitted(model, 'coef_')

        # make the prediction
        n_models = len(self.models)
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict(X) / n_models
        return predictions.astype(int)

    def predict_proba(self, X, y=None):
        """
        Creates the prediction ensemble.

        Parameters:
        -----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The feature matrix to predict, for each instance, the class.

        y : 
            Ignored

        Returns:
        --------
        predictions: array-like
            The predictions of the ensemble of classifiers.

        """
        X = check_array(X)
        # check each model is fitted
        for model in self.models:
            check_is_fitted(model, 'coef_')

        # make the prediction
        n_models = len(self.models)
        predictions = np.zeros(len(X))
        for model in self.models:
            predictions += model.predict_proba(X)[:,1] / n_models
        return predictions