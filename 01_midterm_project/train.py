#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from IPython.display import display
import radiant
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import time
from xgboost import XGBClassifier

# data preparation
df_train_full = pd.read_csv("playground-series-s4e10/train.csv")
df_train_full.columns = df_train_full.columns.str.lower().str.replace(" ", "_")
df_train_full['person_emp_length'] = df_train_full['person_emp_length'].astype('int')
categorical_col = df_train_full.select_dtypes(include='object').columns.to_list()
numerical_col = df_train_full.columns.difference(categorical_col + ['loan_status'] + ['id']).to_list()

def add_features(df):
    df2=df.copy()
    
    df2['total_amount_payable'] = np.floor(df2['loan_amnt'] * (1+df2['loan_int_rate']/100))
    df2['interest'] = np.floor(df2['loan_amnt'] * df2['loan_int_rate']/100)
    df2['credit_hist_ vs_age'] = np.round(df2['cb_person_cred_hist_length'] / df2['person_age'], 2)
    df2['credit_hist_vs_work'] = np.round(df2['person_emp_length']/df2['cb_person_cred_hist_length'], 2)
    df2['risk_flag'] = np.where((df2['cb_person_default_on_file'] == 'Y') & (df2['loan_grade'].isin(['D', 'E', 'F', 'G'])), 1, 0)

    return df2

df_train_full = add_features(df_train_full)
numerical_col_new = df_train_full.columns.difference(categorical_col + ['id']).to_list()

def split_data(df, test_size, random_state):
    df = df_train_full.copy()
    
    df_train, df_val = train_test_split(df, test_size = test_size, random_state = random_state)
    
    y_train = df_train.loan_status.values
    y_val = df_val.loan_status.values

    df_train = df_train.drop(columns  = 'loan_status').reset_index(drop = True)
    df_val = df_val.drop(columns = 'loan_status').reset_index(drop = True)

    return df_train, df_val, y_train, y_val

df_train, df_val, y_train, y_val = split_data(df_train_full, 0.25, 1)

# training 

train_dicts = df_train.to_dict(orient = 'records')
val_dicts = df_val.to_dict(orient = 'records')

dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)



best_params = {
    'colsample_bytree': 0.7796639978116678, 
    'eta': 0.15747194778410148,
    'gamma': 0.053362545117080384, 
    'max_depth': 4, 
    'min_child_weight': 8, 
    'n_estimators': 293,  
    'scale_pos_weight': 6,  
    'subsample': 0.9869606970011566 
}

xgb_final_model = XGBClassifier(
    objective='binary:logistic',
    seed=1,
    **best_params 
)

start_time = time.time()
xgb_final_model.fit(X_train, y_train)
end_time = time.time()

y_pred_prob_xgb = xgb_final_model.predict_proba(X_val)[:, 1]
roc_auc_xgb = roc_auc_score(y_val, y_pred_prob_xgb)

print(f"Total training time for final model: {(end_time - start_time):.3f} seconds")
print(f"ROC AUC on validation set: {roc_auc_xgb:.3f}")

output_file = f'xgb_final_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, xgb_final_model), f_out)

print(f'the model is saved to {output_file}')

