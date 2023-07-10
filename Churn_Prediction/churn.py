import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import seaborn as sb
from tqdm import tqdm
import re
sb.set_style('darkgrid')
rcParams['figure.figsize'] = 8,8
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

train = pd.read_csv('data/Train.csv')
test=  pd.read_csv('data/Test.csv')
test_user_id = np.array(test['user_id'])
train = train.drop(['user_id'],axis=1)
test = test.drop(['user_id'],axis=1)

train['missing_values'] = train.isnull().sum(axis=1).tolist()
test['missing_values'] = test.isnull().sum(axis=1).tolist()

print(train['missing_values'])

full = train.append(test)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
# Initialize the Object
scaler = StandardScaler()
# Fit and Transform The Data

print(len(train)+len(test))
print(len(full))

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')
  
train_packs = imputer.fit_transform(train['TOP_PACK'].to_numpy().reshape(-1,1))
full_packs = imputer.fit_transform(full['TOP_PACK'].to_numpy().reshape(-1,1))
test_packs = imputer.fit_transform(test['TOP_PACK'].to_numpy().reshape(-1,1))

def flat_list(list_array):
    flat_list = [item for sublist in list_array for item in sublist]
    return flat_list    

train['TOP_PACK'] = flat_list(train_packs)
full_top_pack = flat_list(full_packs)

train['REGION'].fillna('NO_REGION',inplace=True)
train['MONTANT']= train['MONTANT'].interpolate(method="linear")
train['FREQUENCE_RECH']= train['FREQUENCE_RECH'].interpolate(method="linear")
train['REVENUE']= train['REVENUE'].interpolate(method="linear")
train['ARPU_SEGMENT']= train['ARPU_SEGMENT'].interpolate(method="linear")
train['FREQUENCE']= train['FREQUENCE'].interpolate(method="linear")

train['DATA_VOLUME'].fillna(0,inplace=True)
train['ON_NET'].fillna(0,inplace=True)
train['FREQ_TOP_PACK'].fillna(0,inplace=True)
train['ORANGE'].fillna(0,inplace=True)
train['TIGO'].fillna(0,inplace=True)
train['ZONE1']. fillna(0,inplace=True)
train['ZONE2'].fillna(0,inplace=True)

test['REGION'].fillna('NO_REGION',inplace=True)
test['MONTANT']= test['MONTANT'].interpolate(method="linear")
test['FREQUENCE_RECH']= test['FREQUENCE_RECH'].interpolate(method="linear")
test['REVENUE']= test['REVENUE'].interpolate(method="linear")
test['ARPU_SEGMENT']= test['ARPU_SEGMENT'].interpolate(method="linear")
test['FREQUENCE']= test['FREQUENCE'].interpolate(method="linear")

test['DATA_VOLUME'].fillna(0,inplace=True)
test['ON_NET'].fillna(0,inplace=True)
test['FREQ_TOP_PACK'].fillna(0,inplace=True)
test['ORANGE'].fillna(0,inplace=True)
test['TIGO'].fillna(0,inplace=True)
test['ZONE1']. fillna(0,inplace=True)
test['ZONE2'].fillna(0,inplace=True)


def tfidf(top_pack_list):
    vectorizer = CountVectorizer(analyzer= 'word', stop_words='english') 
    X = vectorizer.fit_transform(top_pack_list)
    r = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    transformer = TfidfTransformer(smooth_idf=True)
    Z = transformer.fit_transform(X)
    d = pd.DataFrame(Z.toarray(),columns=vectorizer.get_feature_names())
    print(d.tail(20))
    return d

tfidf_full =tfidf(full_top_pack) 

#REGION#
def itemize_regions(df):
    regions = df.groupby(['REGION']).size().reset_index().rename(columns={0:'count'})
    print(regions)
    test_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    modules_list = list(regions['REGION'])
    res = {} 
    for key in modules_list:
        for value in test_values:
            res[key] = value
            test_values.remove(value)
            break 
    return res

res = itemize_regions(train)

train['REGION'] = train.replace({"REGION" : res})
test['REGION'] = test.replace({"REGION" : res})


def zones(df):
    total_zones = []
    for i in tqdm(range(0,len(df))):
        if df['ZONE1'].iloc[i]==0 or  df['ZONE2'].iloc[i]== 0: 
           val1 = 0  
        elif df['MONTANT'].iloc[i] == 0: 
           val1 = 0
        else:
            val1 = df['MONTANT'].iloc[i]/ (df['ZONE1'].iloc[i] + df['ZONE2'].iloc[i])
        total_zones.append(val1)
    print(len(total_zones))
    return total_zones

train['spread_zones'] = zones(train)
test['spread_zones'] = zones(test)

def networks_spread(df):
    total_nets = []
    for i in tqdm(range(0,len(df))):
        if df['ORANGE'].iloc[i] ==0 and  df['TIGO'].iloc[i]== 0.0: 
           val1 = 0  
        else:
            val1 = df['MONTANT'].iloc[i]/(df['ORANGE'].iloc[i] + df['TIGO'].iloc[i])
       
        total_nets.append(val1)
    print(len(total_nets))
    return total_nets


def money_reg(df):
    val1 = []
    val2 =[]
    val3 = []
    for i in tqdm(range(0,len(df))):
        val1.append(df['MONTANT'].iloc[i]/df['REGULARITY'].iloc[i])
        val2.append(df['REVENUE'].iloc[i]/df['REGULARITY'].iloc[i])
        val3.append(df['ARPU_SEGMENT'].iloc[i]/df['REGULARITY'].iloc[i])          
    return val1,val2,val3

train['mon_reg'],train['rev_reg'],train['segment_reg'] = money_reg(train)
test['mon_reg'],test['rev_reg'],test['segment_reg'] = money_reg(test)

train['profit_reg'] = train['rev_reg']-train['mon_reg'] 
test['profit_reg'] = test['rev_reg']-test['mon_reg'] 


print(train[['mon_reg','rev_reg','segment_reg','profit_reg']].head(10))

def networks_total(df):
    total_nets = []
    for i in tqdm(range(0,len(df))):
        if df['ORANGE'].iloc[i]==0 and  df['TIGO'].iloc[i]== 0.0: 
           val1 = 0  
        else:
            val1 = (df['MONTANT'].iloc[i] * df['FREQUENCE_RECH'].iloc[i])/(df['ORANGE'].iloc[i] + df['TIGO'].iloc[i])
        total_nets.append(val1)
    return total_nets

train['spread_nets_total'] = networks_total(train)
test['spread_nets_total'] = networks_total(test)


train['spread_nets'] = networks_spread(train)
test['spread_nets'] = networks_spread(test)

train.loc[~np.isfinite(train['spread_nets']), 'spread_nets'] = 0 
test.loc[~np.isfinite(test['spread_nets']), 'spread_nets'] = 0

print(train[['spread_nets','spread_zones']])

#TOP PACK#
def top_rack_DTYPE(df):
    n_data = []
    for k in tqdm(range(0,len(df))):
        nets = re.findall(r'[O|o]n[- ]net',str(df['TOP_PACK'].iloc[k])) 
        data = re.findall(r'[D|d]ata',str(df['TOP_PACK'].iloc[k])) 
        all_net = re.findall(r'[A|a]ll[- ]net',str(df['TOP_PACK'].iloc[k])) 
            #check if any have len
        if nets:
            n_data.append(1)
        elif data:
            n_data.append(2)
        elif all_net:
            n_data.append(3)
        else:
            n_data.append(0)
    print(len(n_data))
    return n_data

train['DTYPE'] = top_rack_DTYPE(train)
test['DTYPE'] = top_rack_DTYPE(test)


def top_rack(df):
    k_value = []
    for k in tqdm(range(0,len(df))):
        k_value.append(len(str(df['TOP_PACK'].iloc[k])))
    return k_value

def top_rack_days(df):
    n_time = []
    for k in tqdm(range(0,len(df))):
        small_set = re.findall(r'\d+[D|d]',str(df['TOP_PACK'].iloc[k])) 
        if len(small_set) > 0:
            n_time.append(int(str(small_set[0])[:-1]))            
        else:
            n_time.append(0)
    print(len(n_time))
    return n_time


def top_rack_GB(df,d_type):
    n_data = []
    for k in tqdm(range(0,len(df))):
        small_set = re.findall(r'\d+'+d_type,str(df['TOP_PACK'].iloc[k])) 
            #small_set = re.findall(r'\d{1,3}[G|M]B',df['TOP_PACK'].iloc[k]) 
        if len(small_set) > 0:
            n_data.append(int(small_set[0][:-2]))
        else:
            n_data.append(0)
    print(len(n_data))
    return n_data

def top_rack_F(df):
    f1_data = []
    f2_data = []
    f3_data = []
    for k in tqdm(range(0,len(df))):
        small_set = re.findall(r'\d+F',str(df['TOP_PACK'].iloc[k])) 
            #small_set = re.findall(r'\d{1,3}[G|M]B',df['TOP_PACK'].iloc[k]) 
        if len(small_set) == 3:
            f1_data.append(int(small_set[0][:-1]))
            f2_data.append(int(small_set[1][:-1]))
            f3_data.append(int(small_set[2][:-1])) 
        elif len(small_set) == 2:
            f1_data.append(0)   
            f2_data.append(int(small_set[0][:-1]))
            f3_data.append(int(small_set[1][:-1]))
        elif len(small_set) == 1: 
            f1_data.append(0)   
            f2_data.append(0)
            f3_data.append(int(small_set[0][:-1]))
        else:
            f1_data.append(0)
            f2_data.append(0)
            f3_data.append(0)

    print(len(f1_data))
    return f1_data,f2_data,f3_data


train['F1'], train['F2'],train['F3']   = top_rack_F(train) 
test['F1'],test['F2'],test['F3'] = top_rack_F(test) 

train['n_gb'] = top_rack_GB(train,'GB') 
test['n_gb'] = top_rack_GB(test,'GB')

train['n_mb'] = top_rack_GB(train,'MB') 
test['n_mb'] = top_rack_GB(test,'MB')


train['k_len'] = top_rack(train) 
test['k_len'] = top_rack(test)

train['n_days'] = top_rack_days(train) 
test['n_days'] =top_rack_days(test)


#TENURE#
def tenure_featEng(frame):
    key = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5,'I':6,'J':7,'K':8}
    letters = []
    for i in tqdm(range(0,len(frame))):
        letter =key[str(frame.TENURE.iloc[i][0])]
        letters.append(letter)
    return letters

def tenure_months(frame):
    min_months = []
    max_months=[]
    avg_months = [] 
    for k in tqdm(range(0,len(frame))):
        small_set = re.findall(r'\d+',str(frame['TENURE'].iloc[k]))
        if(len(small_set)==2):
            min_months.append(small_set[0])
            max_months.append(small_set[1])
            avg_months.append((int(small_set[1])+int(small_set[0]))/2)
        else:
            min_months.append(0)
            max_months.append(small_set[0])
            avg_months.append(24)
                
    return min_months,max_months,avg_months  
                   
train['tenure_letter'] = tenure_featEng(train)
test['tenure_letter'] = tenure_featEng(test)

train['min_months'], train['max_months'],train['avg_months'] = tenure_months(train)
test['min_months'], test['max_months'],test['avg_months'] = tenure_months(test)

print(train[['min_months','max_months','avg_months']].head())


def dataPerMonth(df,d_type):
    n_min = []
    n_max = []
    
    for k in tqdm(range(0,len(df))):
        if d_type =='GB': 
            if df['min_months'].iloc[k] != 0:
                n_min.append(int(df['n_gb'].iloc[k])/int(df['min_months'].iloc[k]))
            else:
                n_min.append(0)
            
            if  df['max_months'].iloc[k] != 0:
                n_max.append(int(df['n_gb'].iloc[k])/int(df['max_months'].iloc[k]))
            
            else:
                n_max.append(0)
            
        elif d_type =='MB':
            if df['min_months'].iloc[k] != 0: 
                n_min.append(int(df['n_mb'].iloc[k])/int(df['min_months'].iloc[k]))
            else:
                n_min.append(0)
            
            if df['max_months'].iloc[k] != 0:
                n_max.append(int(df['n_mb'].iloc[k])/int(df['max_months'].iloc[k]))
            else:    
                n_max.append(0)
    
    return n_min ,n_max

train['n_gbAvgMin'], train['n_gbAMax']  = dataPerMonth(train,'GB') 
train['n_gbAvgMin'], train['n_gbAMax']  = dataPerMonth(train,'GB') 

train['n_mbAvgMin'], train['n_mbAMax']  = dataPerMonth(train,'MB') 
train['n_mbAvgMin'], train['n_mbAMax']  = dataPerMonth(train,'MB') 


test['n_gbAvgMin'], test['n_gbAMax']  = dataPerMonth(test,'GB') 
test['n_gbAvgMin'], test['n_gbAMax']  = dataPerMonth(test,'GB') 

test['n_mbAvgMin'], test['n_mbAMax']  = dataPerMonth(test,'MB') 
test['n_mbAvgMin'], test['n_mbAMax']  = dataPerMonth(test,'MB') 

print(train[['n_gbAvgMin','n_mbAMax','n_gbAvgMin','n_gbAMax','n_gb','n_mb','min_months','max_months']].head(20))

print(train['TOP_PACK'].head(20))
encoder = LabelEncoder()

train['TOP_PACK'] =encoder.fit_transform(train['TOP_PACK'])
test['TOP_PACK'] =encoder.fit_transform(test['TOP_PACK'])

train['TENURE'] =encoder.fit_transform(train['TENURE'])
test['TENURE'] =encoder.fit_transform(test['TENURE'])

train['freq/rech'] = train['FREQUENCE'] / train['FREQUENCE_RECH']
test['freq/rech'] = test['FREQUENCE'] / test['FREQUENCE_RECH']

train['spend_trend'] = (train['F1']+train['F2']+train['F3'])/train['REGULARITY']
test['spend_trend'] = (test['F1']+test['F2']+test['F3'])/test['REGULARITY']

train['reg_log'] = np.log1p(train['REGULARITY'])
test['reg_log'] = np.log1p(test['REGULARITY'])

num_cols = ['TENURE','TOP_PACK','MONTANT','FREQUENCE_RECH','FREQUENCE','REVENUE','ARPU_SEGMENT','DATA_VOLUME','ON_NET','avg_income','tenure_letter','REGION','k_len','min_months','max_months','arpu_avg','avg_months','usage_amnt','n_gb','n_mb','F1','F2','F3','avg_data','avg_net','REGULARITY','FREQ_TOP_PACK','min_tenure_montant','max_tenure_montant','DTYPE','customer_profit','n_days','total_amnt','arpu_segment','total_income','total_arpu','ORANGE','ZONE1','ZONE2','customer_profit_avg','spread_nets','spread_zones','main_profit','freq_freq_rech','n_gbAvgMin','n_mbAMax','n_gbAvgMin','n_gbAMax','spread_nets_total','freq/rech','segment_reg','mon_reg','rev_reg','profit_reg','spend_trend','missing_values','reg_log']

def tenure_av(df,t_month):
    avg_tenure = []
    for i in tqdm(range(0,len(df))):
        if df['MONTANT'].iloc[i] == 0 or df[t_month].iloc[i] == 0:
            val1 = 0
        else:
            val1 = df['MONTANT'].iloc[i]/df[t_month].iloc[i]
        avg_tenure.append(val1)
    return avg_tenure

##MONTANT###
train['FREQUENCE_RECH'].fillna((train['FREQUENCE_RECH'].mean()), inplace=True)
test['FREQUENCE_RECH'].fillna((test['FREQUENCE_RECH'].mean()), inplace=True)

train['min_months']=train.min_months.astype(float)
train['max_months']=train.max_months.astype(float)
train['MONTANT']=train.MONTANT.astype(float)

test['min_months']=test.min_months.astype(float)
test['max_months']=test.max_months.astype(float)
test['MONTANT']=test.MONTANT.astype(float)

train['min_tenure_montant'] = tenure_av(train,'min_months')
train['max_tenure_montant'] = tenure_av(train,'max_months')

test['min_tenure_montant'] = tenure_av(test,'min_months')
test['max_tenure_montant'] = tenure_av(test,'max_months')

##iNCOME/REVENUE
def money_usage(df):
    avg_money = []
    total_money = []
    for i in tqdm(range(0,len(df))):
        if df['FREQUENCE_RECH'].iloc[i] == 0: 
           val1 = 0
           val2 = 0
        elif df['MONTANT'].iloc[i] == 0: 
           val1 = 0
           val2 = 0
        else:
            val1 = df['MONTANT'].iloc[i]/df['FREQUENCE_RECH'].iloc[i]
            val2 = df['MONTANT'].iloc[i]*df['FREQUENCE_RECH'].iloc[i]
        avg_money.append(val1)
        total_money.append(val2)
    return avg_money,total_money

def avg_total_income(df):
    avg_income = []
    total_income = []
    for i in tqdm(range(0,len(df))):
        if df['FREQUENCE'].iloc[i] == 0: 
           val1 = 0
           val2 = 0
        elif df['REVENUE'].iloc[i] == 0: 
           val1 = 0
           val2 = 0
        else:
            val1 = df['REVENUE'].iloc[i]/df['FREQUENCE'].iloc[i]
            val2 = df['REVENUE'].iloc[i]*df['FREQUENCE'].iloc[i]   
        avg_income.append(val1)
        total_income.append(val2)   
    return avg_income,total_income

def avg_ARPU(df):
    arpu = []
    total_arpu = []
    for i in tqdm(range(0,len(df))):
        if df['ARPU_SEGMENT'].iloc[i] == 0: 
           val1 = 0 
           val2 = 0 
        elif df['FREQUENCE'].iloc[i] == 0: 
           val1 = 0
           val2 = 0
        else:
            val1 = df['ARPU_SEGMENT'].iloc[i]/df['FREQUENCE'].iloc[i]
            val2 = df['ARPU_SEGMENT'].iloc[i]*df['FREQUENCE'].iloc[i]
        arpu.append(val1)
        total_arpu.append(val2)
    return arpu,total_arpu


def freq_freq_rech(df):
    freq = []
    for i in tqdm(range(0,len(df))):
        if df['FREQUENCE'].iloc[i] == 0: 
           val1 = 0  
        elif df['FREQUENCE'].iloc[i] == 0: 
           val1 = 0
        else:
            val1 = df['FREQUENCE_RECH'].iloc[i]*df['FREQUENCE'].iloc[i]
        freq.append(val1)
    return freq

train['freq_freq_rech'] = freq_freq_rech(train)
test['freq_freq_rech']  = freq_freq_rech(test)

def profit_IncomeRevenue(df):
    profit = []
    for i in tqdm(range(0,len(df))):
        val1 = df['total_income'].iloc[i] - df['total_amnt'].iloc[i]
        profit.append(val1)
    return profit


def profit_IncomeRevenueAVG(df):
    profit = []
    for i in tqdm(range(0,len(df))):
        val1 = df['avg_income'].iloc[i] - df['usage_amnt'].iloc[i]
        profit.append(val1)
    return profit


def spread_Montat_networks(df):
    networks = []
    for i in tqdm(range(0,len(df))):
        val2 = df['ORANGE'].iloc[i] + df['TIGO'].iloc[i]
        val1 = df['MONTANT']/val2
        networks.append(val1)
    return networks

def main_profit(df):
    main_profit = []
    for i in tqdm(range(0,len(df))):
        val2 = df['REVENUE'].iloc[i] - df['MONTANT'].iloc[i]
        main_profit.append(val2)
    return main_profit


train['main_profit']= main_profit(train) 
test['main_profit'] = main_profit(test)


train['usage_amnt'], train['total_amnt']= money_usage(train) 
test['usage_amnt'], test['total_amnt'] = money_usage(test)

train['avg_income'], train['total_income']= avg_total_income(train) 
test['avg_income'], test['total_income']= avg_total_income(test)

train['arpu_avg'],train['total_arpu'] = avg_ARPU(train) 
test['arpu_avg'],test['total_arpu'] = avg_ARPU(test)

train['customer_profit'] = profit_IncomeRevenue(train)
test['customer_profit']  = profit_IncomeRevenue(test)

train['customer_profit_avg'] = profit_IncomeRevenueAVG(train)
test['customer_profit_avg']  = profit_IncomeRevenueAVG(test)


###NET AVG######
net_columns = ['DATA_VOLUME','ON_NET','ARPU_SEGMENT']

def avg_data(df,data_columns):
    data_avg = []
    df[data_columns].fillna(0, inplace=True)
    for i in tqdm(range(0,len(df))):
        if df[data_columns].iloc[i] == 0: 
            val1 = 0 
        elif df[data_columns].iloc[i] == 0: 
            val1 = 0 
        else:
            val1 = df[data_columns].iloc[i]/df['REGULARITY'].iloc[i]
        data_avg.append(val1)
    return data_avg


train['avg_data'] = avg_data(train,net_columns[0])
test['avg_data'] = avg_data(test,net_columns[0])

train['avg_net'] = avg_data(train,net_columns[1])
test['avg_net'] = avg_data(test,net_columns[1])

train['arpu_segment'] = avg_data(train,net_columns[2])
test['arpu_segment'] = avg_data(test,net_columns[2])

y = train['CHURN']

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns 

def k_means(dataF):
    scaler.fit(dataF)
    customers_normalized = scaler.transform(dataF)

    K = range(1, 20)
    distortions = []
    inertias = []
    mapping1 = {}
    sse = {}
    for k in tqdm(K):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(customers_normalized)
    
        distortions.append(sum(np.min(cdist(customers_normalized, kmeans.cluster_centers_,
                                        'euclidean'), axis=1)) / customers_normalized.shape[0])
        inertias.append(kmeans.inertia_)
 
        mapping1[k] = sum(np.min(cdist(customers_normalized, kmeans.cluster_centers_,
                                   'euclidean'), axis=1)) / customers_normalized.shape[0]
    
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

        model = KMeans(n_clusters=15, random_state=42)
        model.fit(customers_normalized)
    
    return  model.labels_
    
print(len(tfidf_full.iloc[len(train):]))
tr_df = tfidf_full.iloc[len(train):].reset_index()

t_test=pd.concat([test[num_cols],tr_df],axis=1)
t_test = t_test.drop(columns=['index'])
x =  pd.concat([train[num_cols], tfidf_full.iloc[:len(train)]],axis=1)

x['Clusters'] = k_means(x)
t_test['Clusters'] = k_means(t_test)

x = x[num_cols]
t_test = t_test[num_cols]


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns 


###Backward Elimination
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state=1)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = 0.2,random_state=1)


from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import reverse_geocoder as rg

import time
import datetime
from scipy.signal import argrelextrema
from xgboost import XGBRegressor


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
t_test = scaler.transform(t_test)

datasets = {'x_train': X_train,
            'y_train': y_train,
            'x_val': X_val,
            'y_val': y_val,
            'x_test': X_test,
            'y_test': y_test,
            'predict':t_test,
            }

from catboost import CatBoostRegressor, cv, Pool
cb_learn_rate = 0.006
n_iterations = 80000
early_stop_rounds = 400

opt_catboost_params = {'iterations' : n_iterations,
                       'learning_rate' : cb_learn_rate,
                       'depth': 7,
                       'bootstrap_type' : 'Bernoulli',
                       'random_strength': 1,
                       'min_data_in_leaf': 10,
                       'l2_leaf_reg': 3,
                       'loss_function' : 'RMSE', 
                       'eval_metric' : 'RMSE',
                       'grow_policy' : 'Depthwise',
                       'max_bin' : 1024, 
                       'model_size_reg' : 0,
                       'task_type' : 'GPU',
                       'od_type' : 'IncToDec',
                       'od_wait' : 100,
                       'metric_period' : 500,
                       'verbose' : 500,
                       'subsample' : 0.8,
                       'od_pval' : 1e-10,
                       'max_ctr_complexity' : 8,
                       'has_time': False,
                       'simple_ctr' : 'FeatureFreq',
                       'combinations_ctr': 'FeatureFreq',
                       'random_seed' : 13}


cb_reg = CatBoostRegressor(**opt_catboost_params)

cb_reg.fit(X_train, y_train, eval_set=(X_val, y_val), 
           use_best_model=True,
           plot=True, 
           early_stopping_rounds=early_stop_rounds)

from sklearn import metrics
y_t = np.array(datasets['y_test'])

cat_preds =np.array(cb_reg.predict(datasets['x_test']))


def local_Test(pred,y_values):
    submit_final= []
    fpr, tpr, thresholds = metrics.roc_curve(y_values, pred, pos_label=1)

    super_preds = pd.DataFrame(list(zip(pred, y_values)),
               columns =['Pred', 'Y'])

    return super_preds,metrics.auc(fpr, tpr)

d2,auc2 = local_Test(cat_preds,y_t)

print(d2.head(30))
print(auc2)


submit_final=[]
pred_final = np.array(cb_reg.predict(datasets['predict']))
for j in range(0,len(pred_final)):
    if pred_final[j] < 0:
        submit_final.append(pred_final[j]*(-1))
    else:
        submit_final.append(pred_final[j])

finalsub = pd.DataFrame(list(zip(test_user_id, submit_final)),
               columns =['user_id', 'CHURN'])

finalsub.to_csv('submit93.csv',index = None)
