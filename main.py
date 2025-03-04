import pandas as pd
import os

# Import .py scripts from subdirectories 
from module.config import *
from module.dataset import *
from module.modeling.train import *
from module.features import *

# Check if train.csv and pred.csv exist in user_data folder, otherwise load synthetic datasests
if os.path.exists(user_data_dir_train) and os.path.exists(user_data_dir_pred):
    train_df = pd.read_csv(user_data_dir_train, sep = '\t')
    pred_df = pd.read_csv(user_data_dir_pred, sep = '\t')
    print ('User dataset found and loaded')
else:
    train_df = pd.read_csv(synth_data_dir_train, sep = '\t')
    pred_df = pd.read_csv(synth_data_dir_pred, sep = '\t')
    print ('Pre-uploaded synthetic dataset found and loaded')

# Basic data cleaning: drop rows that are duplicate and change any NA values to the average value of the column it's in. 
train_cleaned = basic_cleaning (train_df)
pred_cleaned = basic_cleaning (pred_df)

# Apply function that changes categorical data into numerical data so it can be used as input for the models 
train_processed, pred_processed = convert_categorical_to_dummies (train_cleaned, pred_cleaned)

# Use the function standardize_min_max to standardize the train and pred datasets using a min max scaler and save them as .csv files in the folder data/interim. These datasets can be used for the lasso regression model, because reggression is sensitive to scaling 
train_df_sdd, pred_df_sdd = standardize_dataset (train_processed, pred_processed)

# Code checks if run_grid_search = True or False in config.py file. If using your own datasets, change run_grid_search in the config.py file to True 
# so the models are trained on your own data. 
if run_grid_search == True:
    best_rf_model = randomforestregressormodel_train(train_processed)
    best_lasso_model = lassoregressionmodel_train(train_df_sdd)
    best_svm_model = supportvectormachinemodel_train(train_processed)
else:
    print("Gridsearch is False in the config.py file, proceeding with the pre-trained models")
# Folds = number of train/test splits of the dataset, candidates = models with different parameters and fits = folds * candidates

# Import code that loads the trained models and that can predict on the dataset
from module.modeling.predict import *

# Use the loaded models to predict on the dataset
ranked_students_rf = randomforestregressormodel_pred (pred_processed)
ranked_students_lasso = lassoregressionmodel_pred(pred_df_sdd, pred_processed)
ranked_students_svm = supportvectormachinemodel_pred(pred_processed)

if save_method == 'xlsx':
    # Save results as excel file in the folder predictions. Predictions is in the models folder.
    writer = pd.ExcelWriter('models/predictions/ranked_students.xlsx', engine='xlsxwriter')
    ranked_students_rf.to_excel(writer, sheet_name='Random Forest', startrow=0, startcol=0, index=False)
    ranked_students_lasso.to_excel(writer, sheet_name='Lasso', startrow=0, startcol=0, index=False)
    ranked_students_svm.to_excel(writer, sheet_name='Support Vector Machine', startrow=0, startcol=0, index=False)
    writer.close()
elif save_method == 'csv':
    # Save results as CSV files
    ranked_students_rf.to_csv('ranked_students_rf.csv', sep='\t', index=False)
    ranked_students_lasso.to_csv('ranked_students_lasso.csv', sep='\t', index=False)
    ranked_students_svm.to_csv('ranked_students_svm.csv', sep='\t', index=False)
else:
    print("Invalid save method. Please choose 'xlsx' or 'csv'.")