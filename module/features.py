from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from module.config import *

# Function that checks the train and the predict dataset for categorical columns. It then creates a new column for each category and drops the first column to prevent multicolinearity. 
# Then the categorized train and predict dataset are aligned to account for all the categories (if the train or predict dataset includes categories not present in the other) 
def convert_categorical_to_dummies(train_dataset, predict_dataset):
    categorical_cols = train_dataset.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        train_dataset = pd.get_dummies(train_dataset, columns=categorical_cols, drop_first=True)
        predict_dataset = pd.get_dummies(predict_dataset, columns=categorical_cols, drop_first=True) 
        train_dataset, predict_dataset = train_dataset.align(predict_dataset, join='outer', axis=1, fill_value=0)   
        dummy_cols = [col for col in train_dataset.columns if any(col.startswith(cat_col) for cat_col in categorical_cols)]
        train_dataset[dummy_cols] = train_dataset[dummy_cols].astype(int)
        predict_dataset[dummy_cols] = predict_dataset[dummy_cols].astype(int)
    train_dataset.to_csv('data/processed/train_processed.csv', sep='\t', index=False) 
    predict_dataset.to_csv('data/processed/pred_processed.csv', sep='\t', index=False) 
    return train_dataset, predict_dataset

# Use min/max scaler for LASSO regression and save tbe datasets in the data/interim folder
def standardize_dataset (dataset_train, dataset_pred):
    column_names_train = dataset_train.columns.tolist()
    column_names_pred = dataset_pred.columns.tolist()
    train_scaled_data = MinMaxScaler().fit_transform(dataset_train)
    pred_scaled_data = MinMaxScaler().fit_transform(dataset_pred)
    train_df_scaled1 = pd.DataFrame(train_scaled_data, columns=column_names_train)
    pred_df_scaled1 = pd.DataFrame(pred_scaled_data, columns=column_names_pred)
    train_df_scaled1.to_csv('data/interim/train_data_standardized.csv', sep='\t', index=False) 
    pred_df_scaled1.to_csv('data/interim/pred_data_standardized.csv', sep='\t', index=False) 
    return train_df_scaled1,  pred_df_scaled1