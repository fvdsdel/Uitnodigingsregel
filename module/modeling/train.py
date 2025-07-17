from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

def save_model(model, model_name):
    """Saves the model to a file."""
    joblib.dump(model, f'models/{model_name}.joblib')

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


def randomforestregressormodel_train(dataset_train, random_seed, dropout_column, model_params):
    """Trains a Random Forest Regressor model."""
    X,y = prep_data(dataset_train, dropout_column)
    estimator_class = RandomForestRegressor
    estimator = estimator_class(random_state=random_seed)
    grid_kwargs = {'refit':True,'n_jobs': -1, 'verbose': 2}
    grid_search = run_grid_search(estimator, X, y, model_params, grid_kwargs)
    best_model = estimator_class(**grid_search.best_params)
    best_model.fit(X, y) 
    save_model(best_model, 'random_forest_regressor')
    return best_model


def lassoregressionmodel_train (dataset_train_sdd, random_seed, dropout_column, alpha_range):
    """Trains a Lasso regression model."""
    X, y = prep_data(dataset_train_sdd, dropout_column)
    estimator_class = Lasso
    estimator = estimator_class(random_state = random_seed)
    model_params = {'alpha':alpha_range}
    grid_kwargs = {'refit': False, 'cv': 5, 'n_jobs': -1, 'verbose': 2}
    grid_search = run_grid_search(estimator, X, y, model_params, grid_kwargs)
    best_model = estimator_class(**grid_search.best_params)
    best_model.fit(X, y) 
    save_model(best_model, 'lasso_regression')
    return best_model


def supportvectormachinemodel_train(dataset_train_sdd, random_seed, dropout_column, model_params):
    """Trains a Support Vector Machine model."""
    X,y = prep_data(dataset_train_sdd, dropout_column)
    estimator_class = SVC
    estimator = estimator_class(random_state=random_seed, probability = True)
    grid_kwargs = {'refit': False, 'n_jobs': -1, 'verbose': 2}
    grid_search = run_grid_search(estimator, X, y, model_params, grid_kwargs)
    best_model = estimator_class(**grid_search.best_params, probability = True)
    best_model.fit(X, y) 
    save_model(best_model, 'support_vector_machine')
    return best_model
