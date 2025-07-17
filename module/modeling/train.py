from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

def save_model(model, model_name):
    """Saves the model to a file."""
    joblib.dump(model, f'models/{model_name}.joblib')

# Random Forest Regressor model for training 
def randomforestregressormodel_train(dataset_train, random_seed, dropout_column, model_params):
    
    X = dataset_train.drop(dropout_column, axis=1).values
    y = dataset_train[dropout_column].values
    estimator_class = RandomForestRegressor
    estimator = estimator_class(random_state=random_seed)
    
    # Hyperparameter tuning using gridsearchCV
    grid_search = GridSearchCV(estimator, model_params, refit = True, n_jobs = -1, verbose = 2)
    grid_search.fit(X, y) 
    best_params = grid_search.best_params_
    best_model = estimator_class(**best_params)
    best_model.fit(X, y) 
    
    # Save model 
    save_model(best_model, 'random_forest_regressor')
    return best_model

# lasso regression model for training 
def lassoregressionmodel_train (dataset_train_sdd, random_seed, dropout_column, alpha_range):
    
    X = dataset_train_sdd.drop(dropout_column, axis=1).values
    y = dataset_train_sdd[dropout_column].values
    estimator_class = Lasso
    estimator = estimator_class(random_state = random_seed)
    param = {'alpha':alpha_range}
    
    # Hyperparameter tuning using gridsearchCV
    grid_search = GridSearchCV(estimator, param_grid = param, refit = False, cv=5, n_jobs = -1, verbose = 2)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = estimator_class(**best_params)
    best_model.fit(X, y) 
    
    # Save model 
    save_model(best_model, 'lasso_regression')
    return best_model

# Support vector machine model for training 
def supportvectormachinemodel_train(dataset_train_sdd, random_seed, dropout_column, model_params):
    X = dataset_train_sdd.drop(dropout_column, axis=1).values
    y = dataset_train_sdd[dropout_column].values
    estimator_class = SVC
    estimator = estimator_class(random_state=random_seed, probability = True)
    # Hyperparameter tuning using gridsearchCV 
    grid_search = GridSearchCV(estimator, model_params, refit = False, n_jobs = -1, verbose = 2) 
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = estimator_class(**best_params, probability = True)
    best_model.fit(X, y) 

    # Save model 
    save_model(best_model, 'support_vector_machine')
    return best_model
