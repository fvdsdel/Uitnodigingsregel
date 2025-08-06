import pandas as pd
import joblib

from .models import ESTIMATORS

def _predict(estimator, X_pred):
    best_model = joblib.load(f'models/{ESTIMATORS[estimator]["NAME"]}.joblib')
    
    if ESTIMATORS[estimator].get("PROB",None):
        yhat2 = best_model.predict_proba(X_pred)
        yhat2_uitval = yhat2[:, 1]
        pred_data0 = pd.DataFrame({'voorspelling': yhat2_uitval})
    else:
        yhat2 = best_model.predict(X_pred)
        pred_data0 = pd.DataFrame({'voorspelling': yhat2})
    return pred_data0


def predict(estimator,pred_df,pred_df2, dropout_column, studentnumber_column):
    """Predict using the specified estimator."""
    if dropout_column in pred_df.columns:
        X_pred = pred_df.drop(dropout_column, axis=1).values
    else:
        X_pred = pred_df.values
    # Save the studentnumbers so they can be matched with the predictions after the model is done
    X_pred_studentnumber = pred_df2[[studentnumber_column]]
    
    pred_data0 = _predict(estimator, X_pred)
    pred_data = pd.concat([pred_data0, X_pred_studentnumber],axis=1).reindex(pred_data0.index)
    pred_data.rename(columns={0: 'StudentNr'}, inplace=True)
    # Sort the predictions based on the predicted value and rearrange the column order. 
    pred_data['ranking'] = pred_data['voorspelling'].rank(method = 'dense', ascending=False)
    pred_data = pred_data.sort_values(by=['voorspelling'], ascending=False).reset_index(drop=True)
    student_ranked_data = pd.DataFrame(data = pred_data)
    student_ranked_data_ordered = student_ranked_data[['ranking', studentnumber_column, 'voorspelling']]
    return student_ranked_data_ordered

def randomforestregressormodel_pred (pred_df, dropout_column, studentnumber_column):
    """Predict using the Random Forest Regressor model."""
    student_ranked_data_ordered = predict('RF', pred_df,pred_df, dropout_column, studentnumber_column)
    return student_ranked_data_ordered

def lassoregressionmodel_pred (pred_df_sdd, dataset_pred, dropout_column, studentnumber_column):
    """Predict using the Lasso Regression model."""
    student_ranked_data_ordered = predict('LASSO', pred_df_sdd,dataset_pred, dropout_column, studentnumber_column)
    return student_ranked_data_ordered

def supportvectormachinemodel_pred (pred_df_sdd, dataset_pred, dropout_column, studentnumber_column):
    """Predict using the Support Vector Machine model."""
    student_ranked_data_ordered = predict('SVM', pred_df_sdd,dataset_pred, dropout_column, studentnumber_column)
    return student_ranked_data_ordered