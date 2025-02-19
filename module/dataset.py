# Function that drops duplicate rows and replaces any NA-values with the average of the column it's in
def basic_cleaning (dataset): 
    dataset_no_dups = dataset.drop_duplicates()
    dataset_cleaned = dataset_no_dups.fillna(dataset_no_dups.mean())
    return dataset_cleaned 