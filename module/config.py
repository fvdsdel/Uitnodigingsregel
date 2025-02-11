from pathlib import Path
import numpy as np

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT/'data'
RAW_DATA_DIR = PROJ_ROOT/DATA_DIR/'raw'
INTERIM_DATA_DIR = PROJ_ROOT/DATA_DIR/'interim'
PROCESSED_DATA_DIR = PROJ_ROOT/DATA_DIR/'processed'
MODELS_DIR = PROJ_ROOT/'models'

#REPORTS_DIR = PROJ_ROOT/'reports'
#FIGURES_DIR = PROJ_ROOT/REPORTS_DIR/'figures'

# Paths for datasets
user_data_dir_train = PROJ_ROOT/DATA_DIR/'raw'/'user_data'/'train.csv'
user_data_dir_pred = PROJ_ROOT/DATA_DIR/'raw'/'user_data'/'pred.csv'
synth_data_dir_train = PROJ_ROOT/DATA_DIR/'raw'/'synth_data_train.csv'
synth_data_dir_pred = PROJ_ROOT/DATA_DIR/'raw'/'synth_data_pred.csv'
standardized_data_train = PROJ_ROOT/DATA_DIR/'interim'/'train_data_standardized.csv'
standardized_data_pred = PROJ_ROOT/DATA_DIR/'interim'/'pred_data_standardized.csv'

# Define the column name that contains dropout (uitval) in the datasets
dropout_column = 'Dropout'

# Define constants for RF, lasso and SVM
random_seed = 42

rf_parameters = {
    'bootstrap': [True, False],
    'max_depth': [2, 3, 4],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3, 5],
    'n_estimators': [100, 200, 300]}

alpha_range = np.arange(0.01, 2, 0.01)

svm_parameters = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}