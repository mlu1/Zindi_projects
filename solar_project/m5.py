import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import  GradientBoostingClassifier,RandomForestClassifier
from tqdm import tqdm
from sklearn.linear_model import  LinearRegression, Ridge
import os, gc, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA, FastICA, TruncatedSVD,FactorAnalysis
from sklearn.mixture import GaussianMixture
pca = PCA(random_state=42,n_components=1)
from sklearn.preprocessing import MinMaxScaler 
import seaborn as sns
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cluster import KMeans
import random
SEED = 2023
random.seed(SEED)
np.random.seed(SEED)


train_path = 'data/Train.csv' # use your path
test_path = 'data/Test.csv'
subs = pd.read_csv('data/SampleSubmission.csv')

t1 = pd.read_csv(train_path)
t2 = pd.read_csv(test_path)

t2_cols = t2.columns
test = t2
train = t1

train=train.fillna(0)
test=test.fillna(0)

city_codes = {'Lagos':0, 'Nairobi':1, 'Bujumbura':2,'Kampala':3}
country_codes = {'Ghana':0, 'Kenya':1, 'Uganda':2, 'Cameroon':3,'Nigeria':4}

train['city'] = train['city'].map(city_codes)
train['country'] = train['country'].map(country_codes)
test['city'] = test['city'].map(city_codes)
test['country'] = test['country'].map(country_codes)


def  time_features(x): 
    x["date"]=pd.to_datetime(x["date"])
    x["dayofweek"]=x["date"].dt.dayofweek
    x["dayofyear"]=x["date"].dt.dayofyear
    x["dayofmonth"]=x["date"].dt.day
    x["hour"]=x["date"].dt.hour
    x["is_weekend"]=x["dayofweek"].apply( lambda x : 1 if x  in [5,6] else 0 )
    x["year"]=x["date"].dt.year
    x["quarter"]=x["date"].dt.quarter
    x["month"]=x["date"].dt.month
    return x

time_cols =['month','quarter','year','is_weekend','hour','dayofmonth','dayofyear','dayofweek'] 
train=time_features(train)
test=time_features(test)


percentage_missing = train.isnull().sum()*100/len(train)
to_drop = percentage_missing[percentage_missing>90].keys()

train.drop(to_drop,axis=1,inplace=True)
train.drop(['date','site_id','id'],axis=1,inplace=True)

test.drop(to_drop,axis=1,inplace=True)
test.drop(['date','site_id','id'],axis=1,inplace=True)

print(train[time_cols])
print(train.columns)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import os

OUTPUT_DIR = 'data/'
N_SPLITS = 10
PARAMS = {
    #'objective': 'regression',
    'boosting': 'gbdt',
    'seed': SEED,
    'num_leaves': 56,
    'learning_rate': 0.035,
    'feature_fraction': 0.4,
    'bagging_fraction': 1.0,
    'n_jobs': -1,
    'lambda_l2': 0.168,
    'lambda_l1': 1.8e-7,
    'verbose': 1,
    'min_data_in_leaf': 20,
    'max_bin': 255,
}


def train_lgbm(df:pd.DataFrame, t_cols:list,output_dir:str,y_col:str):
    """ Train & Save lgbm model and Save feature_importance.csv for each fold """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_index, valid_index) in enumerate(kf.split(df)):
        print(f'label={y_col} fold[{fold}]', flush=True)
        tr_ds = lgb.Dataset(
            df.iloc[train_index][t_cols],
            df.iloc[train_index][y_col]

        )
        val_ds = lgb.Dataset(
            df.iloc[valid_index][t_cols],
            df.iloc[valid_index][y_col],

        )
        clf = lgb.train(
            params=PARAMS,
            train_set=tr_ds,
            num_boost_round=4000000,
            valid_sets=val_ds,
            # feval=calculate_log_loss,
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
        )
        clf.save_model(
            f'{output_dir}/{y_col}_f{fold+1:02d}.lgb',
            num_iteration=clf.best_iteration,
        )
        feature_imp = pd.DataFrame(
            {
                'Feature': t_cols,
                'Value': clf.feature_importance(),
            }
        )
        feature_imp.to_csv(
            f'{output_dir}/{y_col}_f{fold+1:02d}_feature_importance.csv',
            index=False,
        )


def infer_lgbm(df:pd.DataFrame, infer_cols:list, y_col:str, output_dir:str):
    """ Inferring using df(test data) """
    ys = []
    for fold in range(N_SPLITS):
        print(fold, end=', ')
        model_file = f'{output_dir}/{y_col}_f{fold+1:02d}.lgb'
        if not os.path.exists(model_file): break
        clf = lgb.Booster(model_file=model_file)
        y = clf.predict(df[infer_cols], num_iteration=clf.best_iteration)
        ys.append(y)
    return np.mean(ys, axis=0)

columns=['pm2_2']
scaler = StandardScaler()
label_col = 'pm2_5'
tr_df = train


new_list =[]
for item in train.columns:
    if item != 'pm2_5':
        new_list.append(item)

train_cols = new_list



train.replace([np.inf, -np.inf], 0, inplace=True)
train=train.fillna(0)
test.replace([np.inf, -np.inf], 0, inplace=True)
test=test.fillna(0)

print(train[train_cols].head(40))

train_lgbm(tr_df,train_cols,OUTPUT_DIR,'pm2_5')
labels_col = infer_lgbm(test, train_cols, label_col, OUTPUT_DIR)

print(labels_col)

# Make prediction on the test set
new_predicts = []
zero_count = 0
for i in range(0,len(labels_col)):
  if labels_col[i] >=0.5:
    new_predicts.append(1)
  elif labels_col[i] < 0.5:
    new_predicts.append(0)
    zero_count = zero_count+1
  else:
    continue
print('Number ZERO')
print(zero_count)

sub_file = pd.DataFrame({'id':subs['id'],'pm2_5':new_predicts})
print(sub_file.head(20))
df2.to_csv('sub_m20.csv',index=None)
