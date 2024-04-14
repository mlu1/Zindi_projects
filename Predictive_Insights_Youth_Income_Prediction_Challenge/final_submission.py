import numpy as np
import pandas as pd
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
import xgboost as xgb
import catboost as cat_
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.cluster import KMeans
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler()
from sklearn.preprocessing import KBinsDiscretizer
df_train = pd.read_csv("data/Train.csv")
df_test = pd.read_csv("data/Test.csv")
from sklearn.naive_bayes import ComplementNB
dense_bins = 10
strategy = 'kmeans'
seed = 2023
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)
df_train['nulls']=df_train.isnull().sum(axis=1)
df_test['nulls']=df_test.isnull().sum(axis=1)


def func(x):
    try:
        return dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    except:
        return pd.NaT


def age_ranges(data):
    age=[]
    for i in range(0,len(data)):
        x=data['Age'].iloc[i]
        if x < 18:
            age.append(0)
        elif x >= 18 and x < 21: 
            age.append(1)
        elif x >= 21 and x <24:
            age.append(2)
        elif x >= 24 and x <36:
            age.append(3)
        elif x >=36 and x < 56:
            age.append(4)
        elif x >= 56 and x <60:
            age.append(5)
        elif x >= 60:
            age.append(6)
        else:
            age.append(7)
    return age

def sums_def(cols,df):
    row_sum = []
    min_val =[]
    max_val=[]
    medians=[]
    real_median=[]
    for n in tqdm(range(0,len(df))):
        sums=0
        means=0
        min_max_val=[]
        meds=[]
        for col in cols:
            sums += df[col].iloc[n]  
            min_max_val.append(df[col].iloc[n])
        row_sum.append(sums)
        medians.append(sums/len(cols))
        max_val.append(max(min_max_val))
    return row_sum,max_val,medians


df_train['Survey_date'].apply(func)
df_test['Survey_date'].apply(func)

df_train=df_train.fillna(0)
df_test=df_test.fillna(0)


df_train['x_data']= pd.to_datetime(df_train['Survey_date'] ,format='%Y-%m-%d') 
df_train['x_month'] = df_train['x_data'].dt.month.astype(int)
df_train['year'] = df_train['x_data'].dt.year.astype(int)
df_train['x_day'] = df_train['x_data'].dt.day.astype(int)
df_train['s_quarter']=df_train['x_data'].dt.quarter 
df_train['s_weekday'] = (df_train['x_day'] // 5 == 1).astype(float)
df_train['dayofweek'] = df_train['x_data'].dt.dayofweek
df_train['DayOfYear'] = df_train['x_data'].dt.dayofyear
df_train['ismonthstart'] = df_train['x_data'].dt.is_month_start
df_train['ismonthend'] = df_train['x_data'].dt.is_month_end
df_train['dayofweek_name'] = df_train['x_data'].dt.day_name()
df_train['is_weekend'] = np.where(df_train['dayofweek_name'].isin(['Sunday','Saturday']),1,0)


df_test['x_data']= pd.to_datetime(df_test['Survey_date'] ,format='%Y-%m-%d') 
df_test['x_month'] = df_test['x_data'].dt.month.astype(int)
df_test['year'] = df_test['x_data'].dt.year.astype(int)
df_test['x_day'] = df_test['x_data'].dt.day.astype(int)
df_test['s_quarter']=df_test['x_data'].dt.quarter 
df_test['s_weekday'] = (df_test['x_day'] // 5 == 1).astype(float)
df_test['dayofweek'] = df_test['x_data'].dt.dayofweek
df_test['DayOfYear'] = df_test['x_data'].dt.dayofyear
df_test['ismonthstart'] = df_test['x_data'].dt.is_month_start
df_test['ismonthend'] = df_test['x_data'].dt.is_month_end
df_test['dayofweek_name'] = df_test['x_data'].dt.day_name()
df_test['is_weekend'] = np.where(df_test['dayofweek_name'].isin(['Sunday','Saturday']),1,0)


education = ['Matric','Degree','Diploma']
subjects =['Math','Mathlit','Additional_lang','Science'] 


def status_collect(df):
    probs = []
    for i in range(0,len(df)):
        if df['Status'].iloc[i] == 'self employed':
            probs.append(1)     
        elif df['Status'].iloc[i] == 'studying':
            probs.append(0.5)
        elif df['Status'].iloc[i] == 'wage and self employed':
            probs.append(2.5) 
        elif df['Status'].iloc[i] == 'other':
            probs.append(1.5)
        elif df['Status'].iloc[i] == 'employment programme':
            probs.append(1.75) 
        elif df['Status'].iloc[i] == 'unemployed':
            probs.append(0)
        else:
            probs.append(2) 
    return probs

df_train['emp_prob'] = status_collect(df_train)
df_test['emp_prob'] = status_collect(df_test)



def fillSubjects(df,subjs):
    subs_filled = []
    for j in range(0,len(df)):
        test_2 = df[subjs].iloc[j]
        if test_2 != test_2:
            subs_filled.append(-1)
        else:
            subs_filled.append(test_2) 
    return subs_filled

def str_contains(df,sub_jects):   
    over_70=[]
    over_60=[]
    over_50=[]
    over_40=[]
    for j in range(0,len(df)):
        score_70=0
        score_60=0
        score_50=0
        score_40=0
        for cols in sub_jects:
            test_1 = df[cols].iloc[j]
            if str(test_1) in ("70 - 79 %|80 - 100 %"):
                try:
                    value = int(test_1)
                except ValueError:
                     score_70=score_70+1 
            elif str(test_1) in ("50 - 59 %|60 - 69 %"):
                score_60=score_60+1
            elif str(test_1) in ("30 - 39 %|40 - 49 %"):
                score_50=score_50+1 
            elif str(test_1) in ("0 - 29 %"):
                score_40=score_40+1    
            else: 
                continue
        over_70.append(score_70)
        over_60.append(score_60)
        over_50.append(score_50)
        over_40.append(score_40) 

    return over_70,over_60,over_50,over_40
  
df_train['Subjects_over_70'],df_train['Subjects_between_5060'],df_train['Subjects_over_50'],df_train['Subjects_0_29']  = str_contains(df_train,subjects)
df_test['Subjects_over_70'],df_test['Subjects_between_5060'],df_test['Subjects_over_50'],df_test['Subjects_0_29']  = str_contains(df_test,subjects)
print(df_train[['Person_id','Subjects_over_70','Subjects_between_5060','Subjects_over_50','Subjects_0_29']].head(30))

import re
def subject_avg(df,subs):
    c_value=[]
    c_value_min=[]
    c_value_max=[]
    for d in tqdm(range(0,len(df))):
        chars = (str(df[subs].iloc[d]).split('-'))
        if len(chars) <2:
            numb1 = int(chars[0])
            c_value_min.append(0)
            c_value_max.append(0)

        else:
            temp2=int(re.sub(r"[^\w\s]", "", chars[1]))      
            if chars[0] == '':
                temp1=0
            else:
                temp1=int(chars[0])
            numb1=(temp1+temp2)/2
            c_value_min.append(temp1)
            c_value_max.append(temp2)
        c_value.append(numb1)           
    return c_value,c_value_min,c_value_max

train_subs_rows=[]
train_subs_maxs=[]
train_subs_mins=[]


test_subs_rows=[]
test_subs_maxs=[]
test_subs_mins=[]

for c in subjects:
    df_train[c+'avg'],df_train[c+'min'],df_train[c+'max']= subject_avg(df_train,c)  
    train_subs_maxs.append(c+'max')
    train_subs_mins.append(c+'min')


for c in subjects:
    df_test[c+'avg'],df_test[c+'min'],df_test[c+'max'] = subject_avg(df_test,c)     
    test_subs_maxs.append(c+'max')
    test_subs_mins.append(c+'min')

def sums_edu(cols,df):
    row_sum = []
    row_means=[]
    for n in tqdm(range(0,len(df))):
        sums=0
        means=0
        meds=[]
        for col in cols:
            sums += df[col].iloc[n]  
        row_sum.append(sums)
        row_means.append(sums/len(cols))
    return row_sum,row_means

df_train['edu_sum'],df_train['edu_mean'] =sums_edu(education,df_train) 
df_test['edu_sum'],df_test['edu_mean']  =sums_edu(education,df_test) 


def means(cols,df):
    row_means=[]
    for i in range(0,len(df)):
        sums=0
        for col in cols:
            sums += df[col].iloc[i]   
        row_means.append(sums/3)
    return row_means

df_train['subs_means_max'] = means(train_subs_maxs,df_train)
df_test['subs_means_max'] = means(test_subs_maxs,df_test)
  
def score(df):      
    final_score =[]    
    for k in range(0,len(df)):
        if df['subs_means_max'].iloc[k] <=30:
            final_score.append(0)
        elif df['subs_means_max'].iloc[k] >30 and df['subs_means_max'].iloc[k] <= 50 :
            final_score.append(1)
        elif df['subs_means_max'].iloc[k] >50 and df['subs_means_max'].iloc[k] <= 60 :
            final_score.append(2)
        elif df['subs_means_max'].iloc[k] >60  and df['subs_means_max'].iloc[k] <= 70 :
            final_score.append(3)
        elif df['subs_means_max'].iloc[k] >70:
            final_score.append(4)
        else:
            continue    
    return final_score

df_train['final_score'] = score(df_train)
df_test['final_score'] = score(df_test)
      

df_train['subs_means_min'] = means(train_subs_mins,df_train)
df_test['subs_means_min'] = means(test_subs_mins,df_test)


df_train['h_is_null'] = np.where(df_train['Home_lang'].isnull(),   1, 0)
df_test['h_is_null'] = np.where(df_test['Home_lang'].isnull(),   1, 0)

df_train['m_is_null'] = np.where(df_train['Mathlit'].isnull(),   1, 0)
df_test['m_is_null'] = np.where(df_test['Mathlit'].isnull(),   1, 0)


df_train['T/R']=df_train['Tenure']/df_train['Round']
df_train['T/S']=df_train['Tenure']/df_train['Schoolquintile']


df_test['T/R']=df_test['Tenure']/df_test['Round']
df_test['T/S']=df_test['Tenure']/df_test['Schoolquintile']

df_train=df_train.fillna(0)
df_train.replace([np.inf, -np.inf], 0, inplace=True)
df_test=df_test.fillna(0)
df_test.replace([np.inf, -np.inf], 0, inplace=True)


from sklearn import preprocessing
for f in df_test.columns:
    if f in ['Status', 'Geography','Province','Math','Math','Mathlit','Additional_lang','Science']:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))
    else:
        continue

num_cols = ['Tenure']
cat_cols = ['Status', 'Geography','Province','Schoolquintile','Subjects_over_70','Subjects_between_5060','Subjects_over_50','Subjects_0_29','Round']

bin_cols = ['Matric','Degree','Diploma','Female','Sa_citizen']

for col in bin_cols:
    df_train[col] = df_train[col].astype(int)
    df_test[col] = df_test[col].astype(int)


#################################################################
cat_1_features =df_train.filter(cat_cols)
all_cat = cat_1_features

train_pca = pca.fit_transform(all_cat)
df_train['PCA_CAT'] = train_pca[:,0]

cat_2_features =df_test.filter(cat_cols)
all_cat_test = cat_2_features
test_pca = pca.fit_transform(all_cat_test)
df_test['PCA_CAT'] = test_pca[:,0]
###################################################################
bin_features = df_train.filter(bin_cols)
train_pca = pca.fit_transform(bin_features)
df_train['PCA_Bin'] = train_pca[:,0]

bin_test_features = df_test.filter(bin_cols)
test_pca = pca.fit_transform(bin_test_features)
df_test['PCA_Bin'] = test_pca[:,0]

###################################################################

df_train['row_sum_bin'],df_train['row_max_bin'],df_train['row_mean_bin'] = sums_def(bin_cols,df_train)
df_train['row_sum_cat'],df_train['row_max_cat'],df_train['row_mean_cat'] = sums_def(cat_cols,df_train)
df_train['row_sum_nums'],df_train['row_max_nums'],df_train['row_mean_nums']= sums_def(num_cols,df_train)

df_test['row_sum_bin'],df_test['row_max_bin'],df_test['row_mean_bin'] = sums_def(bin_cols,df_test)
df_test['row_sum_cat'],df_test['row_max_cat'],df_test['row_mean_cat'] =sums_def(cat_cols,df_test)
df_test['row_sum_nums'],df_test['row_max_nums'],df_test['row_mean_nums'] = sums_def(num_cols,df_test)

fa = FactorAnalysis(n_components=len(num_cols), rotation='varimax', random_state=0)
fa.fit(df_train[num_cols])
extra_feats = [f'fa_{i}'for i in range(len(num_cols))][:6]

df_train[extra_feats] = fa.transform(df_train[num_cols])[:,:6]
df_test[extra_feats] = fa.transform(df_test[num_cols])[:,:6]


dists = [30, 32, 32, 29, 28, 31, 22, 32, 13, 27, 32, 30, 27, 25] 
dists += [15, 9, 12, 11, 7, 6, 6, 2, 10, 2, 4, 4, 4, 7, 8]

for feature, dist in tqdm(zip(num_cols+cat_cols+bin_cols + extra_feats, dists)):
    #x = train[[feature]].append(test[[feature]])[feature].values.reshape(-1, 1)
    
    train_feature = df_train[[feature]]
    test_feature = df_test[[feature]]
    combined_feature = pd.concat([train_feature, test_feature])
    
    x = combined_feature[feature].values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=dist,
                           max_iter=300,
                           random_state=0).fit(x)    
    clus = pd.get_dummies(gmm.predict(x)).values * x
    clus_train = clus[:len(df_train), :]
    clus_test = clus[len(df_train):, :]
    
    clus_feats = [f'{feature}_gmm_dev_{i}'for i in range(clus_train.shape[1])]
    df_train[clus_feats] = clus_train
    df_test[clus_feats] = clus_test


df_train=df_train.fillna(0)
df_train.replace([np.inf, -np.inf], 0, inplace=True)
df_test=df_test.fillna(0)
df_test.replace([np.inf, -np.inf], 0, inplace=True)


def frequency_encoding(column, df, d_test=None):
    frequencies = df[column].value_counts().reset_index()
    frequencies['index'] = frequencies.index
    
    df_values = df[[column]].merge(frequencies, how='left',
                                   left_on=column, right_on='index').iloc[:,-1].values
    if d_test is not None:
        df_test_values = df_test[[column]].merge(frequencies, how='left',
                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values
    else:
        df_test_values = None
    return df_values, df_test_values


for column in tqdm(cat_cols):
    train_values, test_values = frequency_encoding(column, df_train, df_test)
    df_train[column+'_counts'] = train_values
    df_test[column+'_counts'] = test_values


class CustomEncoder:
    def __init__(self, func=np.mean):
        self.func = func
    
    def transform(self, groupby_col, target_col, data):
        grouped = data.groupby(groupby_col)[target_col]
        if self.func == np.mean:
            result = grouped.mean()
        elif self.func == np.median:
            result = grouped.median()
        elif self.func == np.sum:
            result = grouped.sum()
        elif self.func == np.std:
            result = grouped.std()
        elif self.func == np.min:
            result = grouped.min()
        elif self.func == np.max:
            result = grouped.max()
    
        else:
            raise ValueError("Invalid function")
        return data[groupby_col].map(result)

encoder = CustomEncoder(func=np.sum)
k_sum=0
for n in num_cols: 
    df_train['m_sum'+str(k_sum)+n+cat_cols[2]] = encoder.transform(cat_cols[2], n, df_train)
    df_train['m_sum'+str(k_sum)+n+cat_cols[3]] = encoder.transform(cat_cols[3], n, df_train)
    df_test['m_sum'+str(k_sum)+n+cat_cols[2]] = encoder.transform(cat_cols[2], n, df_test)
    df_test['m_sum'+str(k_sum)+n+cat_cols[3]] = encoder.transform(cat_cols[3], n, df_test)
    k_sum=k_sum+1

encoder = CustomEncoder(func=np.mean)
k_mens=0
for n in num_cols: 
    df_train['m_mean'+str(k_mens)+n+cat_cols[2]] = encoder.transform(cat_cols[2], n, df_train)
    df_train['m_mean'+str(k_mens)+n+cat_cols[3]] = encoder.transform(cat_cols[3], n, df_train)
    df_test['m_mean'+str(k_mens)+n+cat_cols[2]] = encoder.transform(cat_cols[2], n, df_test)
    df_test['m_mean'+str(k_mens)+n+cat_cols[3]] = encoder.transform(cat_cols[3], n, df_test)
    
    k_mens=k_mens+1

def Agg(df,cols1,cols2) :
    for col1 in cols1 :
        for col2 in cols2 :
            df[f"{col1}_{col2}_mean"] = df.groupby(col1)[col2].transform('mean')
            df[f"{col1}_{col2}_nunique"] = df.groupby(col1)[col2].transform('nunique')
    return df

df_train=Agg(df_train,num_cols,cat_cols)
df_test=Agg(df_test,num_cols,cat_cols)


df_train['Age']=df_train['year']-df_train['Birthyear'] 
df_test['Age']=df_test['year']-df_test['Birthyear'] 

df_train['age_size'] = age_ranges(df_train)
df_test['age_size'] = age_ranges(df_test)


df_train['age_tenure']=df_train['Tenure']/df_train['age_size']
df_test['age_tenure']=df_test['Tenure']/df_test['age_size']

df_train['age_s']=df_train['Schoolquintile']/df_train['age_size']
df_test['age_s']=df_test['Schoolquintile']/df_test['age_size']

df_train['S*R']=df_train['Status']*df_train['Round']
df_train['S/R']=df_train['Status']/df_train['Round']
df_test['S*R']=df_test['Status']*df_test['Round']
df_test['S/R']=df_test['Status']/df_test['Round']


df_train=df_train.fillna(0)
df_train.replace([np.inf, -np.inf], 0, inplace=True)
df_test=df_test.fillna(0)
df_test.replace([np.inf, -np.inf], 0, inplace=True)


binary_cross = 'True'
bin_features = bin_cols



dense_features = ['Tenure']
sparse_features = cat_cols

add_sparse_features = []
for fea in tqdm(dense_features, total=len(dense_features)):
    discretizer = KBinsDiscretizer(n_bins=dense_bins, encode='ordinal', strategy=strategy, random_state=seed)
    df_train[fea + '_encode'] = discretizer.fit_transform(np.array(df_train[fea].tolist()).reshape(-1, 1))
    df_test[fea + '_encode'] = discretizer.fit_transform(np.array(df_test[fea].tolist()).reshape(-1, 1))
         

df_train_dummy = df_train.drop(["Person_id", "Survey_date","x_data","dayofweek_name",
"Home_lang","Additional_lang"], axis = 1)
# Convert character variables to dummy variables
#df_train_dummy = pd.get_dummies(df_train_dummy, columns=selected_vars, drop_first=True, dummy_na=True)
df_train_dummy.columns
df_test_dummy = df_test.drop(["Person_id", "Survey_date","x_data","dayofweek_name",
"Home_lang","Additional_lang"], axis = 1)

# Convert character variables to dummy variables
# Clean column names


X = df_train_dummy.drop('Target', axis=1)
y = df_train_dummy['Target']

train_features, valid_features, train_y, valid_y = train_test_split(X, y, test_size = 0.20,random_state = 47)
model = cat_.CatBoostClassifier(random_state = 42)
model.fit(train_features,
        train_y,eval_set = [(train_features, train_y),
        (valid_features,valid_y)],
        early_stopping_rounds = 250, 
        verbose = 200)

feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])
feature_importance_df['feature'] = X.columns
feature_importance_df.sort_values(by='importance', ascending=False, inplace=True)
feat = feature_importance_df[feature_importance_df['importance']>0]

important = feat['feature'].values
X =X[important]
df_test_dummy = df_test_dummy[important] 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUM_FEATURES = len(X.columns)
NUM_CLASSES = 2

class RegressionDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class PredictRegressionDataset(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def loaders(X,V,T,X_y,V_y,T_y):
    train_dataset = RegressionDataset(torch.from_numpy(X).float(), torch.from_numpy(X_y).float())
    val_dataset = RegressionDataset(torch.from_numpy(V).float(), torch.from_numpy(V_y).float())
    test_dataset = RegressionDataset(torch.from_numpy(T).float(), torch.from_numpy(T_y).float())

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)

    return train_val_loader

pred_test = scaler.fit_transform(df_test_dummy)
predict_dataset = PredictRegressionDataset(torch.from_numpy(pred_test).float())
predict_loader = DataLoader(dataset=predict_dataset, batch_size=1)

class BinaryClassification(nn.Module):
    def __init__(self, num_features):
        super(BinaryClassification, self).__init__() 
        self.layer_1 = nn.Linear(num_features, 128)  # Increase hidden units
        self.layer_2 = nn.Linear(128, 128)  # Increase hidden units
        self.layer_out = nn.Linear(128, 1) 
        
        self.relu = nn.LeakyReLU()  # Use LeakyReLU activation function
        self.dropout = nn.Dropout(p=0.2)  # Increase dropout rate
        self.batchnorm1 = nn.BatchNorm1d(128)  # Increase batch size
        self.batchnorm2 = nn.BatchNorm1d(128)  # Increase batch size
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        return x

import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model1 = BinaryClassification(NUM_FEATURES)
model1.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model1.parameters(), lr=LEARNING_RATE)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = (acc * 100)
    return acc

# Define the number of
num_folds = 8
# Define the lioss function and optimizer
loss_fn = criterion
# Define the number of epochs and batch size
num_epochs = 5
# Create the stratified k-fold object
skf = StratifiedKFold(n_splits=num_folds)
trains = scaler.fit_transform(X)
val_deep = np.zeros(X.shape[0])

val_store=[]
temp_store=[]
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    # Define the training and validation sets for this fold
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    X_train =scaler.fit_transform(X_train)
    X_train =np.array(X_train)
    y_train= np.array(y_train)

    X_val =scaler.transform(X_val)
    X_val =np.array(X_val)
    y_val= np.array(y_val)


    # Create the data loaders for this fold
    train_dataset = RegressionDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    val_dataset = RegressionDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
   
    # Train the model for this fold
    for epoch in range(num_epochs):
        model1.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model1(inputs)
            loss = loss_fn(outputs,targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation set
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0
        model1.eval()
        temp_store=[]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model1(inputs)
                val_loss = loss_fn(val_outputs, targets.unsqueeze(1)).item() 
                val_acc = binary_acc(val_outputs, targets.unsqueeze(1)).item()
                num_val_batches += 1
                flat_list = [item for sublist in val_outputs.cpu().numpy() for item in sublist] 
                temp_store.append(flat_list)
            val_store =  [item for sublist in temp_store for item in sublist]
            val_deep[val_idx] = val_store  
        print(f"Fold {fold + 1}, Epoch {epoch + 1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

model1.eval()
y_pred_list = []
with torch.no_grad():
    for X_batch in tqdm(predict_loader):
        X_batch = X_batch.to(device)
        y_test_pred = model1(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_list.append(y_test_pred.cpu().numpy())
test_deep = [a.squeeze().tolist() for a in y_pred_list]


class func() :   
    def __init__(self, train, label, test, model, model_type, random_state):
        self.train, self.label, self.test = train, label, test
        self.model, self.model_type = model, model_type
        self.random_state = random_state
        
        assert self.model_type in ('catboost', 'xgboost', 'lgbm','Gbm'), 'Incorrect model_type'
    def __call__(self, plot = True):
        return self.fit(plot)

    def fit(self, plot):
        def catboost_fit(X_train, X_test, y_train, y_test):
            self.model.fit(X_train,y_train,eval_set=[(X_test,y_test)],early_stopping_rounds=500,
                           verbose=50,use_best_model=True)
            x_test_predict = self.model.predict_proba(X_test)[:,1] 
            x_train_predict = self.model.predict_proba(X_train)[:,1]
            self.val_p[test_index] = x_test_predict
            self.test_p += self.model.predict_proba(self.test)[:,1]
            return x_test_predict, x_train_predict

        def xgboost_fit(X_train, X_test, y_train, y_test):
            self.model.fit(X_train, y_train, early_stopping_rounds = 30, eval_metric="auc",
                           eval_set=[(X_test, y_test)], verbose = True)
            x_test_predict = self.model.predict_proba(X_test, ntree_limit = self.model.get_booster().best_ntree_limit)[:,1]
            x_train_predict = self.model.predict_proba(X_train, ntree_limit = self.model.get_booster().best_ntree_limit)[:,1]
            self.val_p[test_index] = x_test_predict
            self.test_p += self.model.predict_proba(self.test, ntree_limit = self.model.get_booster().best_ntree_limit)[:,1]
            return x_test_predict, x_train_predict

        def lgbm_fit(X_train, X_test, y_train, y_test):
            self.model.fit(X_train, y_train, early_stopping_rounds = 30, eval_metric="auc",
                           eval_set=[(X_test, y_test)], verbose = True)
            x_test_predict = self.model.predict_proba(X_test, num_iteration = self.model.best_iteration_)[:,1]
            x_train_predict = self.model.predict_proba(X_train, num_iteration = self.model.best_iteration_)[:,1]
            self.val_p[test_index] = x_test_predict
            self.test_p += self.model.predict_proba(self.test, num_iteration = self.model.best_iteration_)[:,1]
            return x_test_predict, x_train_predict
        
        def gboost_fit(X_train, X_test, y_train, y_test):
            self.model.fit(X_train, y_train)
            x_test_predict = self.model.predict_proba(X_test)[:,1]
            x_train_predict = self.model.predict_proba(X_train)[:,1]
            self.val_p[test_index] = x_test_predict
            self.test_p += self.model.predict_proba(self.test)[:,1]
            return x_test_predict, x_train_predict
       
        self.val_p = np.zeros(self.train.shape[0])
        mean_val = []
        mean_train = []
        self.test_p = np.zeros(self.test.shape[0])
        splits = 10
        kf = StratifiedKFold(n_splits = splits)
        for fold_count, (train_index, test_index) in enumerate(kf.split(self.train, self.label)):
            X_train,X_test = self.train.iloc[train_index],self.train.iloc[test_index]
            y_train,y_test = self.label.iloc[train_index],self.label.iloc[test_index]

            print(f"================================Fold{fold_count+1}====================================")
            if self.model_type == 'catboost': x_test_predict, x_train_predict = catboost_fit(X_train, X_test, y_train, y_test)
            elif self.model_type == 'xgboost': x_test_predict, x_train_predict = xgboost_fit(X_train, X_test, y_train, y_test)
            elif self.model_type == 'lgbm': x_test_predict, x_train_predict = lgbm_fit(X_train, X_test, y_train, y_test)
            elif self.model_type == 'Gbm': x_test_predict, x_train_predict = gboost_fit(X_train, X_test, y_train, y_test)

            print('\nValidation scores', roc_auc_score(y_test, x_test_predict), log_loss(y_test, x_test_predict))
            print('Training scores', roc_auc_score(y_train, x_train_predict), log_loss(y_train, x_train_predict))
            mean_val.append(roc_auc_score(y_test, x_test_predict))
            mean_train.append(roc_auc_score(y_train, x_train_predict))

        if plot:
            feat_imp = pd.DataFrame(sorted(zip(self.model.feature_importances_,self.train.columns)), columns=['Value','Feature'])
            plt.figure(figsize=(50,50))
            sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))
            plt.ylabel('Feature Importance Score')
            plt.savefig('importance.jpg')
        print(np.mean(mean_val), np.mean(mean_train), np.std(mean_val))
        return self.val_p, self.test_p/splits, self.model


sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1)


params_all_feat = {'depth': 5, 'iterations': 633, 'learning_rate': 0.11345058095313602, 
                   'l2_leaf_reg': 5.249467944690077, 'boosting_type': 'Ordered', 'silent': True}

catboost3 = cat_.CatBoostClassifier(**params_all_feat)

func_= func(X, y, df_test_dummy, catboost3, 'catboost', 1000)
val_p10, test_p10, model10 = func_()

rf_model2 = RandomForestClassifier(max_depth=10,
                                   min_samples_split=10,
                                   min_samples_leaf=15,
                                   n_estimators=400,
                                   n_jobs=-1,
                                   random_state=34,
                                   verbose=True)

func_= func(X, y, df_test_dummy, rf_model2, 'Gbm', 1000)
val_p9, test_p9, mode9 = func_()


catboost2 = cat_.CatBoostClassifier(random_seed=34,
                                bootstrap_type='Bayesian',
                                max_depth=6,
                                learning_rate=0.007,
                                iterations=8000,
                                silent=True,
                                eval_metric='AUC')

func_= func(X, y, df_test_dummy, catboost2, 'catboost', 1000)
val_p8, test_p8, model8 = func_()


gbm_model = GradientBoostingClassifier(max_depth=4,
                                       min_samples_leaf=10,
                                       n_estimators=200,
                                       learning_rate=0.1,
                                       min_samples_split=10,
                                       random_state=10)

func_= func(X, y, df_test_dummy, gbm_model, 'Gbm', 1000)
val_p7, test_p7, model7 = func_()

rf_model = RandomForestClassifier(
        #n_jobs=-1,
        criterion='entropy',
        min_samples_split=10,
        n_estimators=400,
        verbose=True,
        random_state=99
        )

func_= func(X, y, df_test_dummy, rf_model, 'Gbm', 1000)
val_p6, test_p6, model5 = func_()


clf_xgb =xgb.XGBClassifier(max_depth=8,
    learning_rate=0.1,
    n_estimators=1000,
    verbosity=0,
    silent=None,
    objective='binary:logistic',
    booster='gbtree',
    n_jobs=-1,
    nthread=None,
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=0.7,
    colsample_bytree=1,
    colsample_bylevel=1,
    colsample_bynode=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    random_state=0,
    seed=None,)


func_ = func(X,y,df_test_dummy,clf_xgb,'xgboost',1000) 
val_p4, test_p4, model4 = func_()
catboost = cat_.CatBoostClassifier(random_seed=34,
                                    n_estimators=10000,
                                    max_depth=6,
                                    eval_metric='AUC',
                                    reg_lambda = 370)

func_= func(X, y, df_test_dummy, catboost, 'catboost', 1000)
val_p1, test_p1, model1 = func_()

xgboost = xgb.XGBClassifier(objective ='binary:logistic', 
                          eta = 0.99,
                          max_depth = 6, 
                          n_estimators = 5000,
                          reg_lambda = 500,
                          sub_sample = 0.8,
                          colsample_bytree = 0.8,
                          random_state=34)

func_= func(X, y, df_test_dummy, xgboost, 'xgboost', 1000)
val_p2, test_p2, model2 = func_()


print(len(val_p2))
print((test_p2))
print('###########################')
print(len(val_p1))
print(test_p1)
print('###########################')
print('##########################')
print(len(val_p4))
print(test_p4)


from sklearn.linear_model import  LinearRegression, Ridge, Lasso
stack = np.column_stack((val_deep,val_p1,val_p2,val_p4,val_p6,val_p7,val_p8,val_p9,val_p10))
stack_p = np.column_stack((test_deep,test_p1,test_p2,test_p4,test_p6,test_p7,test_p8,test_p9,test_p10))
predict = LinearRegression().fit(stack, y).predict(stack_p)


df_submission = pd.DataFrame({"ID": df_test["Person_id"], "Target": predict})
print(df_submission.head(30))
df_submission.to_csv("submission.csv", index=False)

