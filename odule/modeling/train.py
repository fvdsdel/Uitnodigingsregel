from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from module.config import *

# Random Forest Regressor model for training 
def randomforestregressormodel_train(dataset_train):
    X = dataset_train.drop("Dropout", axis=1).values
    y = dataset_train.Dropout.values
    
    rf = RandomForestRegressor(random_state=RANDOM_SEED)

    grid_model = GridSearchCV(rf, rf_parameters, n_jobs = -1, verbose = 3)
    grid_model.fit(X, y) 
    best_params = grid_model.best_params_
    best_rf_model = RandomForestRegressor(**best_params)
    best_rf_model.fit(X, y) 
    return best_rf_model


