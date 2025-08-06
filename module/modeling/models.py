from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.svm import SVC

ESTIMATORS = {
    'RF': {
        "NAME": "random_forest_regressor"
        ,"CLS": RandomForestRegressor
        ,"GKWARGS": {'refit':True,'n_jobs': -1, 'verbose': 2}
        },
    'LASSO': {
        "NAME": "lasso_regression"
        ,"CLS": Lasso
        ,"GKWARGS": {'refit': False, 'cv': 5, 'n_jobs': -1, 'verbose': 2} 
        },
    'SVM': {
        "NAME": "support_vector_machine"
        ,"CLS": SVC
        ,"GKWARGS": {'refit': False, 'n_jobs': -1, 'verbose': 2} 
        ,"PROB": True
        },
}