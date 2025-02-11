from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from module.config import *

### Use min/max scaler for LASSO regression and save tbe datasets in the data/interim folder
def standardize_dataset (dataset_train, dataset_pred):
    column_names_train = dataset_train.columns.tolist()
    column_names_pred = dataset_pred.columns.tolist()
    train_scaled_data = MinMaxScaler().fit_transform(dataset_train)
    pred_scaled_data = MinMaxScaler().fit_transform(dataset_pred)
    train_df_scaled1 = pd.DataFrame(train_scaled_data, columns=column_names_train)
    pred_df_scaled1 = pd.DataFrame(pred_scaled_data, columns=column_names_pred)
    train_df_scaled1.to_csv('data/interim/train_data_standardized.csv', sep='\t', index=False) 
    pred_df_scaled1.to_csv('data/interim/pred_data_standardized.csv', sep='\t', index=False) 
    return 
