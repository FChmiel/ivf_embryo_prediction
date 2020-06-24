"""
Functions used to develop models in ivf_embryo_prediction

ivf_embryo_prediction, Machine-learnt models for predicting chance
of suitable embryo for D5 transfer or freezing.

© University of Southampton, IT Innovation Centre, 2020

Copyright in this software belongs to University of Southampt
University Road, Southampton, SO17 1BJ, UK.

This software may not be used, sold, licensed, transferred, copied
or reproduced in whole or in part in any manner or form or in or
on any media by any person other than in accordance with the terms
of the Licence Agreement supplied with the software, or otherwise
without the prior written consent of the copyright owners.

This software is distributed WITHOUT ANY WARRANTY, without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE, except where stated in the Licence Agreement supplied with
the software.

Created Date :          24-06-2020

Created for Project :   Fertility predict

Author: Francis P. Chmiel

Email: F.P.Chmiel@soton.ac.uk
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.model_selection import (KFold,
                                     GridSearchCV,
                                     train_test_split)

from sklearn.metrics import roc_auc_score


def create_training_test(df, features=None, seed=123, test_size=0.2):
    """
    Creates the training and test set for modelling. The training at test sets are stratified by age.
    
    Parameters:
    -----------
    df : pd.DataFrame, 
        The HFEA IVF dataset.
    
    features : List[str] (default=None),
        List of features (columns) to include in the training/test sets.
        If None 2 features (age and number eggs collected) are used.

    seed : int (default=123), 
        Random seed for the split.
    
    test_size : float (default=0.2), 
        Size of the hold-out test, as a fraction of the total number of 
        instances.
    
    Returns:
    --------
    X : np.array,
        Training set (shape n x len(features)).
    
    Xt : np.array,
        Hold-out test set (shape k x len(features)).
    
    y : np.array, 
        Labels (length n array) for the training set.
    
    yt : np.array,
        Labels (length k array) for the hold-out test set.
    """
    if features is None:
        features = ['Patient Age at Treatment', 'Fresh Eggs Collected']
    data = df[features].values
    target = df['target'].values
    g = df['Patient Age at Treatment'] # groups used to stratify
    return train_test_split(data, 
                            target,
                            stratify=g,
                            test_size=test_size,
                            random_state=seed)

def fit_model(X, y, model=XGBClassifier, params={}, uid=None,
              cv_splitter=KFold, nsplits=5):
    """
    Fits a model using given params and CV method and returns the fitted 
    classifier. If multiple-fold cross-validation is used each individual 
    classifier is returned within a list.
    
    Parameters:
    -----------
    X : np.array,
        Training data shape (n, m) where n is the number of instances and m the 
        number of features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    model : sklearn.base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
    
    params : dict,
        Parameter dictionary to be passed to the model.
    
    uid : np.array (default=None),
        User ids for each training instance, shape (n,). This is passed to the 
        cv_splitter so (for example) patient-wise validation folds can be 
        created. Only relevant for cv-splitters which use groups 
        (e.g., GroupKFold).
    
    cv_splitter : cross-validation generator (default=KFold), 
        The cross-validation splitter (consitent with sklearn API)
        
    nsplits : int (default=5),
        The number of validation folds to create. Passed to cv_splitter.
        
    Returns:
    --------
    models, List[sklearn.base.BaseEstimater] 
        Returns the fitted model(s) instance.
    """
    
    splitter = cv_splitter(n_splits=nsplits)
    
    oof_preds = np.zeros(len(X))
    models = []
    for train_index, test_index in splitter.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, _ = y[train_index], y[test_index]
        clf = model(**params)
        clf.fit(X_train, y_train)
        models.append(clf)
        oof_preds[test_index] = clf.predict_proba(X_test)[:,1]
    oof_score = roc_auc_score(y, oof_preds)
    return models, oof_score


def tune_model(X, y, param_grid, uid=None, model=XGBClassifier, 
                   cv_splitter=KFold, n_splits=5, metric='roc_auc'):
    """Tunes hyperparameters of a model by performing cross-validated 
    grid_search.
    
    Parameters:
    -----------
    X : np.array, 
        Training data shape (n,m ) where n is the number of instances and m the
        number of features.
        
    y : np.array,
        Labels for training data, shape (n,).
    
    param_grid : dict, 
        Parameter dictionary to be passed to the model. Each value should be a 
        list of values of the specified parameter to search. See Sci-kit Learn
        documentation for more information.
            
    model : sklearn.Base.BaseEstimater (default=XGBClassifier),
        The classifier to fit. Must be compatible with the sklearn API.
        
    cv_splitter : cross-validation generator (default=GroupKFold),
        The cross-validation splitter (consitent with sklearn API).
    
    nsplits : int,
        The number of validation folds to create. Passed to cv_splitter.
        
    Returns
    --------
    clf : GridSearchCV,
        A fitted instance of a GridSearchCV class.
        
    """
    # instantiate model
    clf = XGBClassifier()
    # get cv_splits idxs
    cv_splitter = cv_splitter(n_splits=n_splits)
    cv_splits = []
    for train_index, test_index in cv_splitter.split(X, y):
        cv_splits.append((train_index, test_index))
    hyperparameter = GridSearchCV(estimator=clf, param_grid=param_grid, 
                                  scoring=metric, cv=cv_splits)
    hyperparameter.fit(X, y)
    return hyperparameter