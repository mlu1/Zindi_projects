import pandas as pd 
import numpy as np 
import datetime as dt #used for datatime datatype 
import seaborn as sns #used for visualization and EDA 
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score, confusion_matrix, log_loss, classification_report, make_scorer, roc_curve, roc_auc_score
from lightgbm import LGBMClassifier #algorithm  for classification 
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold #used for cross validation 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold#used for cross validation by kfold 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

train = pd.read_csv('data/Train.csv')
test=  pd.read_csv('data/Test.csv')
sub=pd.read_csv('data/sample.csv')
states = pd.read_csv('data/NigerianStateNames.csv')
test_user_id = np.array(test['ID'])

test['State'].fillna('No State',inplace=True)
train['State'].fillna('No State',inplace=True)

train['Car_Category'].fillna('NO_CAT',inplace=True)
test['Car_Category'].fillna('NO_CAT',inplace=True)



train['LGA_Name'].fillna('NO_LGA',inplace=True)
test['LGA_Name'].fillna('NO_LGA',inplace=True)

test['Subject_Car_Colour'].fillna('NO_Color',inplace=True)
train['Subject_Car_Colour'].fillna('NO_Color',inplace=True)

test['Gender'].fillna('Gender',inplace=True)
train['Gender'].fillna('Gender',inplace=True)

test['ProductName'].fillna('ProductName',inplace=True)
train['ProductName'].fillna('ProductName',inplace=True)





train['Policy Start Date']=pd.to_datetime(train['Policy Start Date'])
train['Policy End Date']=pd.to_datetime(train['Policy End Date'])
train['First Transaction Date']=pd.to_datetime(train['First Transaction Date'])

train['s_yr']=train['Policy Start Date'].dt.year
train['s_month']=train['Policy Start Date'].dt.month
train['s_day']=train['Policy Start Date'].dt.day
train['s_quarter']=train['Policy End Date'].dt.quarter 
train['s_weekday'] = (train['s_day'] // 5 == 1).astype(float)
train['coop_time'] = (max(train['s_yr']) - train['s_yr'])*12 - train['s_month']

train['t_yr']=train['First Transaction Date'].dt.year
train['t_month']=train['First Transaction Date'].dt.month
train['t_day']=train['First Transaction Date'].dt.day
train['t_quarter']=train['First Transaction Date'].dt.quarter 
train['t_weekday'] = (train['t_day'] // 5 == 1).astype(float)
train['coop_time_t'] = (max(train['t_yr']) - train['t_yr'])*12 - train['t_month']
train['e_yr']=train['Policy End Date'].dt.year
train['e_month']=train['Policy End Date'].dt.month
train['e_day']=train['Policy End Date'].dt.day
train['e_quarter']=train['Policy Start Date'].dt.quarter 
train['e_weekday'] = (train['e_day'] // 5 == 1).astype(float)
train['coop_time_e'] = (max(train['e_yr']) - train['e_yr'])*12 - train['e_month']


test['Policy Start Date']=pd.to_datetime(test['Policy Start Date'])
test['Policy End Date']=pd.to_datetime(test['Policy End Date'])
test['First Transaction Date']=pd.to_datetime(test['First Transaction Date'])

test['s_yr']=test['Policy Start Date'].dt.year
test['s_month']=test['Policy Start Date'].dt.month
test['s_day']=test['Policy Start Date'].dt.day
test['s_quarter']=test['Policy Start Date'].dt.quarter 
test['s_weekday'] = (test['s_day'] // 5 == 1).astype(float)
test['coop_time'] = (max(test['s_yr']) - test['s_yr'])*12 - test['s_month']

test['e_yr']=test['Policy End Date'].dt.year
test['e_month']=test['Policy End Date'].dt.month
test['e_day']=test['Policy End Date'].dt.day
test['e_quarter']=test['Policy End Date'].dt.quarter 
test['e_weekday'] = (test['e_day'] // 5 == 1).astype(float)

test['t_yr']=test['First Transaction Date'].dt.year
test['t_month']=test['First Transaction Date'].dt.month
test['t_day']=test['First Transaction Date'].dt.day
test['t_quarter']=test['First Transaction Date'].dt.quarter 
test['t_weekday'] = (test['t_day'] // 5 == 1).astype(float)
test['coop_time_t'] = (max(test['t_yr']) - test['t_yr'])*12 - train['t_month']
test['coop_time_e'] = (max(test['e_yr']) - test['e_yr'])*12 - test['e_month']


train['PolicyDuration_W']=round((train['Policy End Date']-train['Policy Start Date'])/np.timedelta64(1, 'W'))
test['PolicyDuration_W']=round((test['Policy End Date']-test['Policy Start Date'])/np.timedelta64(1, 'W'))


def age_ranges(data):
    age=[]
    for i in range(0,len(data)):
        x=data['Age'].iloc[i]
        if x < 18:
            age.append('young')
        elif x >= 18 and x <36:
            age.append('young Adult')
        elif x >=36 and x < 56:
            age.append('Adult')
        elif x >= 56 and x <59:
            age.append('senior')
        elif x >= 60:
            age.append('retired') 
        else:
            age.append('unkown')
    
    return age 


def age_pol(data):
    s1=[]
    s2=[]
    for i in range(0,len(data)):
       s1.append(data['Age'].iloc[i]/data['No_Pol'].iloc[i])
       s2.append(data['Age'].iloc[i]*data['No_Pol'].iloc[i])
        
    return s1,s2 


train['Age'] = np.where(train['Age'].between(-6099,1), np.NaN, train['Age'])
test['Age'] = np.where(test['Age'].between(-6099,1), np.NaN, test['Age'])
y=train['target'] 
train['age_size'] = age_ranges(train)
test['age_size'] = age_ranges(test)

train['s1'],train['s2'] = age_pol(train)
test['s1'],test['s2'] = age_pol(test)

train = train.fillna(0)
test  = test.fillna(0)
train.replace([np.inf, -np.inf], 0, inplace=True)
test.replace([np.inf, -np.inf], 0, inplace=True)

col_to_transform =['Car_Category','Subject_Car_Colour','Gender','State','LGA_Name','ProductName','age_size','car_type']

col_to_drop = ['Policy End Date', 'Policy Start Date','ID','target','First Transaction Date']

for col in col_to_drop:
    if col in train.columns:
        train.drop([col], axis=1, inplace=True)
    if col in test.columns:
        test.drop([col], axis=1, inplace=True)


from sklearn import preprocessing
for f in test.columns:
    if train[f].dtype=='object' or test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


model = LGBMClassifier(learning_rate=0.22,
        n_estimators = 96,
        cat_smooth=10,
        metrics='binary_error',
        scale_pos_weight= 3.6,
        max_depth=16,
        num_leaves=49,
        reg_lambda=0.3)


s_fold =StratifiedKFold(n_splits=10, shuffle=True,random_state=2021)
scores = []  
preds= [] 
i = 1
for tr, tst in tqdm(s_fold.split(train, y)):
    x_train, x_test=train.iloc[tr],train.iloc[tst]
    y_train, y_test =y.iloc[tr],y.iloc[tst]  
    model.fit(x_train, y_train)
    score = f1_score(y_test,model.predict(x_test))
    pred =model.predict_proba(test)[:,1] 
    scores.append(score)
    preds.append(pred)
    i += 1

final_preds = np.mean(preds, axis=0)
print(np.mean(scores))
final = []  
submit_final=[]

for j in range(0,len(final_preds)):
    if final_preds[j] >= 0.51:
        submit_final.append(1)
    else:
        submit_final.append(0)

finalsub = pd.DataFrame(list(zip(test_user_id, submit_final)),
               columns =['ID','target'])

finalsub.to_csv('final.csv',index = None)
