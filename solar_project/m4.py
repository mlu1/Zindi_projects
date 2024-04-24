import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from lightgbm import LGBMRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder


def eval_metrics(y_ture,y_hat):
    return mean_absolute_error(y_ture,y_hat)


LE = LabelEncoder()
train = pd.read_csv('data/Train.csv')
test = pd.read_csv('data/Test.csv')

city_codes = {'Lagos':0, 'Nairobi':1, 'Bujumbura':2,'Kampala':3}
country_codes = {'Ghana':0, 'Kenya':1, 'Uganda':2, 'Cameroon':3,'Nigeria':4}

train['city'] = train['city'].map(city_codes)
train['country'] = train['country'].map(country_codes)
test['city'] = test['city'].map(city_codes)
test['country'] = test['country'].map(country_codes)

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

test['s_yr']=test['date'].dt.year
test['s_month']=test['date'].dt.month
test['s_day']=test['date'].dt.day

train['s_yr']=train['date'].dt.year
train['s_month']=train['date'].dt.month
train['s_day']=train['date'].dt.day

percentage_missing = train.isnull().sum()*100/len(train)
to_drop = percentage_missing[percentage_missing>90].keys()

train.drop(to_drop,axis=1,inplace=True)
test.drop(to_drop,axis=1,inplace=True)

sample_submission = pd.read_csv('data/SampleSubmission.csv')
train_num_df = train.select_dtypes(include=['number'])
X = train_num_df.drop('pm2_5', axis = 1)
y = train.pm2_5

test_df = test[X.columns]
feautres_name=X.columns
keys= ["id"]

Target_name = y
from xgb import * 
from lgb_model import *

params= {
            'min_child_weight': 10, 
            'eta': 0.004, 
            'colsample_bytree': 0.7, 
            'max_depth': 9,
            'subsample': 0.9, 'lambda': 5, 
            'nthread': 8,
            'n_jobs':-1,
            'booster' : 'gbtree', 
            'eval_metric': 'rmse', 
            'objective': 'reg:linear',
            "gamma":0.5 ,
            "alpha":0.04,
            'silent': 1
        }



lgb_params = {
    'metric': 'rmse',            
    'n_jobs':-1,
    'reg_alpha': 9.271004546600699,
    'reg_lambda': 0.0010084442599664978,
    'colsample_bytree': 0.3,
    'subsample': 0.7,
    'learning_rate': 0.08,
    'max_depth': 20,
    'num_leaves': 20,
    'min_child_samples': 143,
}


lgb_set= lightgbm_model( Train_df=X,
                        Test_df=test_df,
                        feval_metrics=eval_metrics,
                        Target_name=Target_name,
                        params=lgb_params,
                        feature_names=feautres_name,
                        keys=keys,
                        verbose_eval=100,
                        early_stopping_rounds=100,
                        num_boost_round=20000,
                        maximize=False,
                        test_size=0.1,
                        nbr_fold=10,
                        nbr_run=3
                    )

XGBoost= Xgboost_model( Train_df=X,
                        Test_df=test_df,
                        feval_metrics=eval_metrics,
                        Target_name=Target_name,
                        params=params,
                        feature_names=feautres_name,
                        keys=keys,
                        verbose_eval=100,
                        early_stopping_rounds=100,
                        num_boost_round=20000,
                        maximize=False,
                        test_size=0.1,
                        nbr_fold=10,
                        nbr_run=3)


train_pred_lgb_model,test_pred_lgb=lgb_set.lightgbm_Kfold()
test_pred_lgb[Target_name] = test_pred_lgb[Target_name]

train_pred_xgboost,test_pred_xgboost=XGBoost.Xgboost_Kfold()
test_pred_xgboost[Target_name]=test_pred_xgboost[Target_name]

labels_lgb = test_pred_lgb[Target_name]

# Make prediction on the test set
labels_xgb = test_pred_xgboost[Target_name]

final_data_rate = (labels_lgb+labels_xgb)/2

print(final_data_rate)
sub_file = pd.DataFrame({'ID':test_subs,'Yield':final_data_rate})
#final_sub = sub_file.drop(['ID'], axis = 1)

print(sub_file.head(60))
sub_file.to_csv('data_3.csv',index=None)


