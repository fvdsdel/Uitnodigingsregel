# Import packages 
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
else:
    train_df = pd.read_csv(synth_data_dir_train, sep = '\t')
    pred_df = pd.read_csv(synth_data_dir_pred, sep = '\t')

# Use the function standardize_min_max to standardize the train and pred datasets using a min max scaler and save them as .csv files in the folder data/interim. These datasets can be used for the lasso regression model, because reggression is sensitive to scaling 
standardize_dataset (train_df, pred_df)

# Load the standardized datasets
train_df_sdd = pd.read_csv(standardized_data_train, sep = '\t')
pred_df_sdd = pd.read_csv(standardized_data_pred, sep = '\t')

# Run the models with GridsearchCV for optimization and save the fitted models in the folder '/models/'
best_rf_model = randomforestregressormodel_train (train_df)
best_lasso = lassoregressionmodel_train (train_df_sdd)
best_svm_model = supportvectormachinemodel_train (train_df)

# Import code that loads the trained models and that can predict on the dataset
from module.modeling.predict import *

# Use the loaded models to predict on the dataset
ranked_students_rf = randomforestregressormodel_pred (pred_df)
ranked_students_lasso = lassoregressionmodel_pred(pred_df_sdd)
ranked_students_svm = supportvectormachinemodel_pred(pred_df)

# Save results as excel files
writer = pd.ExcelWriter('data/processed/ranked_students.xlsx', engine='xlsxwriter')
ranked_students_lasso.to_excel(writer, sheet_name='Lasso', startrow=0, startcol=0, index=False)
ranked_students_rf.to_excel(writer, sheet_name='Random Forest', startrow=0, startcol=0, index=False)
ranked_students_svm.to_excel(writer, sheet_name='Support Vector Machine', startrow=0, startcol=0, index=False)
writer.close()