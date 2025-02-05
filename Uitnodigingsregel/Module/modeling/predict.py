import pandas as pd
from pathlib import Path
import joblib
import os
from module.config import *

best_rf_model = joblib.load('models/random_forest_regressor.joblib')

# Load Random Forest Regressor model for predictions
def randomforestregressormodel_pred (pred_df, random_state = RANDOM_SEED):
    X_pred = pred_df.values 
    X_pred_studentnumber = pred_df[['Studentnummer']]
    # Add columns for specific Power-BI output
    yhat2 = best_rf_model.predict(X_pred)
    pred_data0 = pd.DataFrame({"yhat2": yhat2})
    pred_data = pd.concat([pred_data0, X_pred_studentnumber],axis=1).reindex(pred_data0.index)
    pred_data.rename(columns={0: 'StudentNr'}, inplace=True)
    
    # Sort results
    pred_data['ranking'] = pred_data['yhat2'].rank(method = 'dense', ascending=False)
    pred_data = pred_data.sort_values(by=['yhat2'], ascending=False).reset_index(drop=True)
    student_ranked_data = pd.DataFrame(data = pred_data)
    student_ranked_data = student_ranked_data.rename({'yhat2': 'voorspelling'}, axis=1)
    student_ranked_data_ordered = student_ranked_data[['ranking', 'Studentnummer', 'voorspelling']]
    return student_ranked_data_ordered
