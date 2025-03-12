from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from module.config import *

# Function that checks the train and the predict dataset for categorical columns. It then creates a new column for each category and drops the first column to prevent multicolinearity. 
# Then the categorized train and predict dataset are aligned to account for all the categories (if the train or predict dataset includes categories not present in the other) 
def convert_categorical_to_dummies(train_dataset:pd.DataFrame, predict_dataset:pd.DataFrame):
    categorical_cols = train_dataset.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        train_dataset = pd.get_dummies(train_dataset, columns=categorical_cols, drop_first=True, dummy_na=True)
        predict_dataset = pd.get_dummies(predict_dataset, columns=categorical_cols, drop_first=True, dummy_na=True) 
        train_dataset, predict_dataset = train_dataset.align(predict_dataset, join='outer', axis=1, fill_value=0)   
        dummy_cols = [col for col in train_dataset.columns if any(col.startswith(cat_col) for cat_col in categorical_cols)]
        train_dataset[dummy_cols] = train_dataset[dummy_cols].astype(int)
        predict_dataset[dummy_cols] = predict_dataset[dummy_cols].astype(int)
    train_dataset.to_csv('data/processed/train_processed.csv', sep='\t', index=False) 
    predict_dataset.to_csv('data/processed/pred_processed.csv', sep='\t', index=False) 
    return train_dataset, predict_dataset

# Use min/max scaler for LASSO regression and save tbe datasets in the data/interim folder
### Add min/max scaler for LASSO regression
def _standardize_ds(df:pd.DataFrame):
    """Use minmax scaler to standardize the dataset
    Args:
        df (pd.DataFrame): The dataset to be standardized
        Returns:
        pd.DataFrame: The standardized dataset
    """
    dropout = df[dropout_column]
    df = df.drop(columns=[dropout_column])
    column_names = df.columns.tolist()
    scaled_data = MinMaxScaler().fit_transform(df)
    df_scaled = pd.DataFrame(scaled_data, columns=column_names)
    df_scaled[dropout_column] = dropout.values
    return df_scaled

def standardize_dataset( dataset_train:pd.DataFrame, dataset_pred:pd.DataFrame):
    train_df_scaled = _standardize_ds(dataset_train)
    pred_df_scaled = _standardize_ds(dataset_pred)
    train_df_scaled.to_csv('data/interim/train_data_standardized.csv', sep='\t', index=False)
    pred_df_scaled.to_csv('data/interim/pred_data_standardized.csv', sep='\t', index=False)
    return train_df_scaled, pred_df_scaled