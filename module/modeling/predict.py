import pandas as pd
import joblib
from module.config import *

best_rf_model = joblib.load('models/random_forest_regressor.joblib')
best_lasso_model = joblib.load('models/lasso_regression.joblib')
best_svm_model = joblib.load('models/support_vector_machine.joblib')

# Load Random Forest Regressor model for predictions
def randomforestregressormodel_pred (pred_df, random_state = random_seed):
    # Check if predict dataset has a dropout column, then convert predict data to values
    if dropout_column in pred_df.columns:
        X_pred = pred_df.drop(dropout_column, axis=1).values
    else:
        X_pred = pred_df.values
    # Save the studentnumbers so they can be matched with the predictions after the model is done
    X_pred_studentnumber = pred_df[[studentnumber_column]]

    # Run model on X_pred and make predictions, rename the new column. Match predictions with the student number. 
    yhat2 = best_rf_model.predict(X_pred)
    pred_data0 = pd.DataFrame({'voorspelling': yhat2})
    pred_data = pd.concat([pred_data0, X_pred_studentnumber],axis=1).reindex(pred_data0.index)
    pred_data.rename(columns={0: 'StudentNr'}, inplace=True)
    
    # Sort the predictions based on the predicted value and rearrange the column order. 
    pred_data['ranking'] = pred_data['voorspelling'].rank(method = 'dense', ascending=False)
    pred_data = pred_data.sort_values(by=['voorspelling'], ascending=False).reset_index(drop=True)
    student_ranked_data = pd.DataFrame(data = pred_data)
    student_ranked_data_ordered = student_ranked_data[['ranking', studentnumber_column, 'voorspelling']]
    return student_ranked_data_ordered

# Lasso regression model for predicitng
def lassoregressionmodel_pred (pred_df_sdd, dataset_pred, random_state = random_seed):
    # Check if standardized predict dataset has a dropout column, then convert this dataset to values 
    if dropout_column in pred_df_sdd.columns:
        X_pred = pred_df_sdd.drop(dropout_column, axis=1).values
    else:
        X_pred = pred_df_sdd.values
    # Save the studentnumbers from the regular predict dataset so they can be matched with the predictions after the model is done
    X_pred_studentnumber = dataset_pred[[studentnumber_column]]

    # Run model on X_pred and make predictions, rename the new column. Match predictions with the student number.
    yhat2 = best_lasso_model.predict(X_pred)
    pred_data0 = pd.DataFrame({'voorspelling': yhat2})
    pred_data = pd.concat([pred_data0, X_pred_studentnumber], axis= 1).reindex(pred_data0.index)

    # Sort the predictions based on the predicted value and rearrange the column order.
    pred_data['ranking'] = pred_data['voorspelling'].rank(method = 'dense', ascending=False)
    pred_data = pred_data.sort_values(by=['voorspelling'], ascending=False).reset_index(drop=True)
    student_ranked_data = pd.DataFrame(data = pred_data)
    student_ranked_data_ordered = student_ranked_data[['ranking', studentnumber_column, 'voorspelling']]
    return student_ranked_data_ordered

# Support Vector Machines for predicting
def supportvectormachinemodel_pred (pred_df_sdd, dataset_pred, random_state = random_seed):
    # Check if standardized predict dataset has a dropout column, then convert this dataset to values
    if dropout_column in pred_df_sdd.columns:
        X_pred = pred_df_sdd.drop(dropout_column, axis=1).values
    else:
        X_pred = pred_df_sdd.values  
    # Save the studentnumbers from the regular predict dataset so they can be matched with the predictions after the model is done
    X_pred_studentnumber = dataset_pred[[studentnumber_column]]

    # Run model on X_pred and make predictions, rename the new column. Match predictions with the student number.
    yhat2 = best_svm_model.predict_proba(X_pred)
    yhat2_uitval = yhat2[:, 1]
    pred_data0 = pd.DataFrame({'voorspelling': yhat2_uitval})
    pred_data = pd.concat([pred_data0, X_pred_studentnumber], axis=1).reindex(pred_data0.index)

    # Sort the predictions based on the predicted value and rearrange the column order.
    pred_data['ranking'] = pred_data['voorspelling'].rank(method = 'dense', ascending=False)
    pred_data = pred_data.sort_values(by=['voorspelling'], ascending=False).reset_index(drop=True)
    student_ranked_data = pd.DataFrame(data = pred_data)
    student_ranked_data_ordered = student_ranked_data[['ranking', studentnumber_column, 'voorspelling']]
    return student_ranked_data_ordered