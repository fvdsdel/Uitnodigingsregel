from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from pandas import DataFrame
import joblib

from .models import ESTIMATORS

def prep_data(dataset_train,dropout_column):
    """Prepares the training data by dropping the dropout column."""
    X = dataset_train.drop(dropout_column, axis=1).values
    y = dataset_train[dropout_column].values
    return X, y

def run_grid_search(estimator, X, y, model_params, grid_kwargs) -> GridSearchCV:
    """Runs a grid search to find the best hyperparameters for the model."""
    grid_search = GridSearchCV(estimator, model_params, **grid_kwargs)
    grid_search.fit(X, y)
    return grid_search

def _train_model(estimator_class:BaseEstimator, dataset_train:DataFrame, random_seed:int,target_column:str,model_params:dict,grid_kwargs:dict,probability=None):
    """Train model with best params trough gridsearch Return refitted model"""
    X,y = prep_data(dataset_train, target_column)
    estimator_params = {}
    if probability:
        estimator_params["probability"] = probability
        print(estimator_params)
    estimator = estimator_class(random_state=random_seed,**estimator_params)
    grid_search = run_grid_search(estimator, X, y, model_params, grid_kwargs)
    best_model = estimator_class(**grid_search.best_params_,**estimator_params)
    best_model.fit(X, y) 
    return best_model

def train_model(esimator_code, dataset_train:DataFrame, random_seed:int,target_column:str,model_params:dict):
    """Train model with best params trough gridsearch Return refitted model"""
    if not (_est:=ESTIMATORS.get(esimator_code,None)):
        raise ValueError(f"Estimator code '{esimator_code}' is not recognized. Available options are: {list(ESTIMATORS.keys())}")
    best_model = _train_model(_est["CLS"],dataset_train,random_seed,target_column=target_column,model_params=model_params,grid_kwargs=_est["GKWARGS"],probability=_est.get("PROB", None)) 
    joblib.dump(best_model, f'models/{_est["NAME"]}.joblib')
    return best_model

def randomforestregressormodel_train(dataset_train, random_seed, dropout_column, model_params):
    """Trains a Random Forest Regressor model."""
    best_model = train_model("RF",dataset_train,random_seed,target_column=dropout_column,model_params=model_params)
    return best_model

def lassoregressionmodel_train (dataset_train, random_seed, dropout_column, alpha_range):
    """Trains a Lasso regression model."""
    model_params = {'alpha':alpha_range}
    best_model = train_model("LASSO",dataset_train,random_seed,target_column=dropout_column,model_params=model_params)
    return best_model

def supportvectormachinemodel_train(dataset_train, random_seed, dropout_column, model_params):
    """Trains a Support Vector Machine model."""
    best_model = train_model("SVM",dataset_train,random_seed,target_column=dropout_column,model_params=model_params)
    return best_model