import pandas as pd
import os

## Import .py scripts from subdirectories 

# Import config file that contains settings that can be adjusted
from module.config import *
# Import script with basic data cleaning: drop duplicate rows and change NA values of numerical columns to the column mean
from module.dataset import *
# Import script that contains 3 models that train on the train dataset. Tuning is done with GridsearchCV. The models are Random Forest (RF),
# Lasso Regression (lasso) and Support Vector Machines (SVM)
from module.modeling.train import *
# Import python script that contains feature engineering. The first function checks a dataset for categorical columns and changes them
# using dummy variables. The second function standardizes the data using a minmax scaler. This is needed for the lasso regression model
# and could in theory help with the accuracy of SVM
from module.features import *

# Check if train.csv and pred.csv exist in user_data folder, otherwise load synthetic datasests. When using user_data, change the name of the 
# train dataset to train.csv and the predict dataset to pred.csv 
# When user data is loaded and an error occurs here, please check if the sep = '\t' needs to be changed to another seperator like ',' or '.'
# This should be the same as the seperator used in the user .csv file and can needs to be changed in the config.py file.
if os.path.exists(user_data_dir_train) and os.path.exists(user_data_dir_pred):
    train_df = pd.read_csv(user_data_dir_train, sep = seperator)
    pred_df = pd.read_csv(user_data_dir_pred, sep = seperator)
    print ('User dataset found and loaded')
else:
    train_df = pd.read_csv(synth_data_dir_train, sep = seperator)
    pred_df = pd.read_csv(synth_data_dir_pred, sep = seperator)
    print ('Pre-uploaded synthetic dataset found and loaded')

# Basic data cleaning: drop rows that are duplicate and change any NA values to the average value of the column it's in. 
train_basic_cl = basic_cleaning (train_df)
pred_basic_cl = basic_cleaning (pred_df)

# Detect if there are columns in which all rows have the same value and delete these columns from the train and predict datasets 
train_cleaned, pred_cleaned = remove_single_value_columns (train_basic_cl, pred_basic_cl)

# Apply function that changes categorical data into numerical data so it can be used as input for the models 
train_processed, pred_processed = convert_categorical_to_dummies (train_cleaned, pred_cleaned)

# Use the function standardize_min_max to standardize the train and pred datasets using a min max scaler and save them as .csv files in the folder data/interim 
# These datasets can be used for the lasso and svm models, because reggression is sensitive to scaling 
train_df_sdd, pred_df_sdd = standardize_dataset (train_processed, pred_processed)

# Code checks if retrain_models = True or False in config.py file. If using your own datasets, change retrain_models in the config.py file to True, so the models are trained on your own data. 
# Warning: training the models can take a long time depending on the size and contents of your data
if retrain_models == True:
    print ('Training models on the data...')
    best_rf_model = randomforestregressormodel_train(train_processed)
    best_lasso_model = lassoregressionmodel_train(train_df_sdd)
    best_svm_model = supportvectormachinemodel_train(train_df_sdd)
else:
    print('retrain_models is False in the config.py file, loading the pre-trained models')

# Folds = number of train/test splits of the dataset, candidates = models with different parameters and fits = folds * candidates

# Import code that loads the trained models and that can predict on the dataset
from module.modeling.predict import *

# Use the loaded models to predict on the datasets. The lasso model uses the standardized dataset ot predict an, but takes the student numnbers from the 
# regular predict dataset. 
ranked_students_rf = randomforestregressormodel_pred (pred_processed)
ranked_students_lasso = lassoregressionmodel_pred(pred_df_sdd, pred_processed)
ranked_students_svm = supportvectormachinemodel_pred(pred_df_sdd, pred_processed)

# Save the output files as either .xlsx or as three .csv files 
if save_method == 'xlsx':
    writer = pd.ExcelWriter('models/predictions/ranked_students.xlsx', engine='xlsxwriter')
    ranked_students_rf.to_excel(writer, sheet_name='Random Forest', startrow=0, startcol=0, index=False)
    ranked_students_lasso.to_excel(writer, sheet_name='Lasso', startrow=0, startcol=0, index=False)
    ranked_students_svm.to_excel(writer, sheet_name='Support Vector Machine', startrow=0, startcol=0, index=False)
    writer.close()
    print ('Output file saved as .xlsx in the /models/predictions folder')
elif save_method == 'csv':
    ranked_students_rf.to_csv('models/predictions/ranked_students_rf.csv', sep='\t', index=False)
    ranked_students_lasso.to_csv('models/predictions/ranked_students_lasso.csv', sep='\t', index=False)
    ranked_students_svm.to_csv('models/predictions/ranked_students_svm.csv', sep='\t', index=False)
    print ('Output files saved as .csv in the /models/predictions/csv_output folder')
else:
    print('Invalid save method. Please choose "xlsx" or "csv".')