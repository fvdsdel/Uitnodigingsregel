# Function that drops duplicate rows and replaces any NA-values with the average of the column if the column contains numerical data
def basic_cleaning(dataset):
    dataset_no_dups = dataset.drop_duplicates()
    numerical_cols = dataset_no_dups.select_dtypes(include=['number'])
    dataset_no_dups.loc[:, numerical_cols.columns] = numerical_cols.fillna(numerical_cols.mean())
    return dataset_no_dups