# Import packages 
import pandas as pd
import os
import yaml

# Choose which settings to load
settings_type = 'default'  # Change to 'custom' to load custom settings

# Load config.yaml file 
config_file = 'config.yaml'

def load_settings(config_file, settings_type = settings_type):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if settings_type == 'default':
        settings = config['default_settings']
    elif settings_type == 'custom':
        settings = config['custom_settings']
    else:
        raise ValueError("No settings found. Choose 'default' or 'custom'.")
    return settings
settings = load_settings(config_file, settings_type)

# Apply settings dynamically
globals().update(settings)

## Import .py scripts from subdirectories 
# Import script with basic data cleaning: drop duplicate rows and change NA values of numerical columns to the column mean
from module.dataset import *
# Import script that contains 3 models that train on the train dataset. Tuning is done with GridsearchCV. The models are Random Forest (RF),
# Lasso Regression (lasso) and Support Vector Machines (SVM)
from module.modeling.train import *
# Import python script that contains feature engineering. The first function checks a dataset for categorical columns and changes them
# using dummy variables. The second function standardizes the data using a minmax scaler. This is needed for the lasso regression model
# and could in theory help with the accuracy of SVM
from module.features import *

# Check if train.csv and pred.csv exist in user_data folder, otherwise load synthetic datasests
# When user data is loaded and an error occurs here, please check if the sep = '\t' needs to be changed in the config.yaml file to another separator like ',' or '.'
# This should be the same as the separator used in your .csv file
if os.path.exists(user_data_dir_train) and os.path.exists(user_data_dir_pred):
    train_df = pd.read_csv(user_data_dir_train, sep = separator, engine='python') 
    pred_df = pd.read_csv(user_data_dir_pred, sep = separator, engine='python') 
    print ('User datasets found and loaded')
else:
    train_df = pd.read_csv(synth_data_dir_train, sep = separator, engine='python')
    pred_df = pd.read_csv(synth_data_dir_pred, sep = separator, engine='python')
    print ('Pre-uploaded synthetic datasets found and loaded')

# Basic data cleaning: drop rows that are duplicate and change any NA values to the average value of the column it's in. 
train_basic_cl = basic_cleaning (train_df)
pred_basic_cl = basic_cleaning (pred_df)

# Detect if there are columns in which all rows have the same value and delete these columns from the train and predict datasets 
train_cleaned, pred_cleaned = remove_single_value_columns (train_basic_cl, pred_basic_cl)

# Apply function that changes categorical data into numerical data so it can be used as input for the models 
train_processed, pred_processed = convert_categorical_to_dummies (train_cleaned, pred_cleaned, dropout_column, separator)

# Use the function standardize_min_max to standardize the train and pred datasets using a min max scaler and save them as .csv files in the folder data/interim 
# These datasets can be used for the lasso and svm models, because reggression is sensitive to scaling 
train_df_sdd, pred_df_sdd = standardize_dataset (train_processed, pred_processed, dropout_column, separator)

# Code checks if retrain_models = True or False in config.yaml file. When using your own datasets, change retrain_models in the config.yaml file to True, so the models are trained on your own data. 
# Warning: training the models can take a long time depending on the size and contents of your data. 
if retrain_models == True:
    print ('Training models on the data...')
    best_rf_model = randomforestregressormodel_train(train_processed, random_seed, dropout_column, rf_parameters)
    best_lasso_model = lassoregressionmodel_train(train_df_sdd, random_seed, dropout_column, alpha_range)
    best_svm_model = supportvectormachinemodel_train(train_df_sdd, random_seed, dropout_column, svm_parameters)
else:
    print('retrain_models is False in the config.yaml file, loading the pre-trained models')
# Folds = number of train/test splits of the dataset, candidates = models with different parameters and fits = folds * candidates

# Import code that loads the trained models and that can predict on the dataset
from module.modeling.predict import *

# Use the loaded models to predict on the datasets. The lasso and SVM models use the standardized dataset ot predict an, but take the student numnbers from the 
# regular predict dataset. 
ranked_students_rf = randomforestregressormodel_pred (pred_processed, dropout_column, studentnumber_column)
ranked_students_lasso = lassoregressionmodel_pred(pred_df_sdd, pred_processed, dropout_column, studentnumber_column)
ranked_students_svm = supportvectormachinemodel_pred(pred_df_sdd, pred_processed, dropout_column, studentnumber_column)

# Save the output files as either .xlsx or as three .csv files 
if save_method == 'xlsx':
    writer = pd.ExcelWriter('models/predictions/ranked_students.xlsx', engine='xlsxwriter')
    ranked_students_rf.to_excel(writer, sheet_name='Random Forest', startrow=0, startcol=0, index=False)
    ranked_students_lasso.to_excel(writer, sheet_name='Lasso', startrow=0, startcol=0, index=False)
    ranked_students_svm.to_excel(writer, sheet_name='Support Vector Machine', startrow=0, startcol=0, index=False)
    writer.close()
    print ('Output file saved as .xlsx in the /models/predictions folder')
elif save_method == 'csv':
    ranked_students_rf.to_csv('models/predictions/csv_output/ranked_students_rf.csv', sep='\t', index=False)
    ranked_students_lasso.to_csv('models/predictions/csv_output/ranked_students_lasso.csv', sep='\t', index=False)
    ranked_students_svm.to_csv('models/predictions/csv_output/ranked_students_svm.csv', sep='\t', index=False)
    print ('Output files saved as .csv in the /models/predictions/csv_output folder')
else:
    print('Invalid save method. For save_method in the config.yaml file, please fill in "xlsx" or "csv"')