from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT/'data'
RAW_DATA_DIR = PROJ_ROOT/DATA_DIR/'raw'
INTERIM_DATA_DIR = PROJ_ROOT/DATA_DIR/'interim'
PROCESSED_DATA_DIR = PROJ_ROOT/DATA_DIR/'processed'
MODELS_DIR = PROJ_ROOT/'models'
REPORTS_DIR = PROJ_ROOT/'reports'
FIGURES_DIR = PROJ_ROOT/REPORTS_DIR/'figures'

# Define paths for datasets
user_data_dir_train = PROJ_ROOT/DATA_DIR/'raw'/'user_data'/'train.csv'
user_data_dir_pred = PROJ_ROOT/DATA_DIR/'raw'/'user_data'/'pred.csv'
synth_data_dir_train = PROJ_ROOT/DATA_DIR/'raw'/'synth_data_train.csv'
synth_data_dir_pred = PROJ_ROOT/DATA_DIR/'raw'/'synth_data_pred.csv'


# Define other constants
RANDOM_SEED = 42
TEST_SIZE = 0.2

rf_parameters = {
    'bootstrap': [True, False],
    'max_depth': [2, 3, 4],
    'max_features': [3, 4, 5],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [2, 3, 5],
    'n_estimators': [100, 200, 300]}

lasso_parameters = {'alpha': [0.0001, 0.005, 0.02, 0.1, 1, 2]}

svm_parameters = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}