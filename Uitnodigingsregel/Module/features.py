from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from module.config import *


if os.path.exists(user_data_dir_train) and os.path.exists(user_data_dir_pred):
    train_df = pd.read_csv(user_data_dir_train, sep = '\t')
    pred_df = pd.read_csv(user_data_dir_pred, sep = '\t')
else:
    train_df = pd.read_csv(synth_data_dir_train, sep = '\t')
    pred_df = pd.read_csv(synth_data_dir_pred, sep = '\t')

## Add min/max scaler for LASSO regression
def standardize_min_max (dataset_train, dataset_pred):
    train_scaled_data = MinMaxScaler().fit_transform(dataset_train)
    pred_scaled_data = MinMaxScaler().fit_transform(dataset_pred)
    return train_scaled_data, pred_scaled_data

train_df_scaled, pred_df_scaled = standardize_min_max(train_df, pred_df)

# Output currenctly are numpy arrays, change to be able to save 

## Store processed dataset in Uitnodigingsregel/data/interim/
# train_df_scaled.to_csv('data/interim/train_scaled.csv', sep='\t', index=False) 
# pred_df_scaled.to_csv('data/interim/pred_scaled.csv', sep='\t', index=False) 