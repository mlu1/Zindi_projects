import pandas as pd
import numpy as np

train_data = pd.read_csv('data/Train.csv')
test_data = pd.read_csv('data/Test.csv')
Samplesub = pd.read_csv('data/SampleSubmission.csv');Samplesub.head()



from math import sin, cos, sqrt, atan2, radians
import time
import datetime

train_data['A'], train_data['B'] = train_data['ward'].str.split(':', 1).str
test_data['A'], test_data['B'] = test_data['ward'].str.split(':', 1).str


def distance_km(lat1,lon1,lat2,lon2):
# approximate radius of earth in km
    R = 6373.0

    lt1 = radians(lat1)
    ln1 = radians(lon1)
    lt2 = radians(lat2)
    ln2 = radians(lon2)

    dlon = ln2 - ln1
    dlat = lt2 - lt1

    a = sin(dlat / 2)**2 + cos(lt1) * cos(lt2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_km = R * c
    
    return distance_km


def data_input(file_name):
    train = pd.read_csv('data/'+file_name)
    return train


def geo_input(file_name):
    data = pd.read_csv(file_name)
    return data


def rotation(data):
  '''
  # most frequently used degrees are 30,45,60
  input: dataframe containing Latitude(x) and Longitude(y)
  '''
  rot_45_x = (0.707 * data['lat']) + (0.707 * data['lon'])
  rot_45_y = (0.707 * data['lon']) + (0.707 * data['lat'])
  rot_30_x = (0.866 * data['lat']) + (0.5 * data['lon'])
  rot_30_y = (0.866 * data['lon']) + (0.5 * data['lat'])
  return rot_45_x, rot_45_y, rot_30_x, rot_30_y


def bearing_degree(lat1, lng1, lat2, lng2):
    '''
    calculate angle between two points
    '''
    radius = 6371  # Mean radius of Earth
    diff_lng = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(diff_lng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(diff_lng)
    return np.degrees(np.arctan2(y, x))


from sklearn.cluster import KMeans
def cluster(data):
  '''
  input: dataframe containing Latitude(x) and Longitude(y) coordinates
  output: series of cluster labels that each row of coordinates belongs to.
  '''
  model = KMeans(n_clusters=50)
  labels = model.fit_predict(data)
  return labels


def haversine_dist(lat1,lng1,lat2,lng2):
  lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
  radius = 6371  # Earth's radius taken from google
  lat = lat2 - lat1
  lng = lng2 - lng1
  d = np.sin(lat/2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng/2) ** 2
  h = 2 * radius * np.arcsin(np.sqrt(d))
  
  return h    


def sine_distance(df,lat,lon):
    
    h_distance = []
    for index, row in df.iterrows():
        distance = haversine_dist(row['lat'],row['lon'],lat,lon)
        h_distance.append(distance)
    
    return h_distance


def manhattan_dist(lat1, lng1, lat2, lng2):
    '''
    calculating two haversine distances by,
     - avoiding Latitude of one point 
     - avoiding Longitude of one point
    and adding it together.
    '''
    a = haversine_dist(lat1, lng1, lat1, lng2)
    b = haversine_dist(lat1, lng1, lat2, lng1)
    return a + b

def m_sine_distance(df,lat,lon):
    m_distance = []
    for index, row in df.iterrows():
        distance = manhattan_dist(row['lat'],row['lon'],lat,lon)
        m_distance.append(distance)
    
    return m_distance


def b_sine_distance(df,lat,lon):    
    b_distance = []
    for index, row in df.iterrows():
        distance = bearing_degree(row['lat'],row['lon'],lat,lon)
        b_distance.append(distance)
    
    return b_distance



steps = 0.2
bins = lambda x: np.floor(x / steps) * steps
test_data["latbin"] = test_data.lat.map(bins)
test_data["lonbin"] = test_data.lon.map(bins)

groups = test_data.groupby(["latbin", "lonbin"])
steps = 0.2
bins = lambda x: np.floor(x / steps) * steps
train_data["latbin"] = train_data.lat.map(bins)
train_data["lonbin"] = train_data.lon.map(bins)
groups = train_data.groupby(["latbin", "lonbin"])



unique_lon_tr =train_data.lon.unique() 
unique_lat_tr =train_data.lat.unique()

unique_lon_ts =test_data.lon.unique() 
unique_lat_ts =test_data.lat.unique()


max_lon_tr = max(unique_lon_tr)
max_lat_tr = max(unique_lat_tr)


max_lon_ts = max(unique_lon_ts)
max_lat_ts = max(unique_lat_ts)


min_lon_tr = min(unique_lon_tr)
min_lat_tr = min(unique_lat_tr)

min_lon_ts = min(unique_lon_ts)
min_lat_ts = min(unique_lat_ts)


all_max_km_tr = []
all_min_km_tr = []

all_max_km_ts = []
all_min_km_ts = []



for index,row in train_data.iterrows():
    max_kms_tr = distance_km(row['lon'],row['lat'],max_lat_tr,max_lon_tr)
    min_kms_tr = distance_km(row['lon'],row['lat'],min_lat_tr,min_lon_tr)    
 
    all_max_km_tr.append(max_kms_tr)
    all_min_km_tr.append(min_kms_tr)

for index,row in test_data.iterrows():
    max_kms_ts = distance_km(row['lon'],row['lat'],max_lat_ts,max_lon_ts)
    min_kms_ts = distance_km(row['lon'],row['lat'],min_lat_ts,min_lon_ts)    
 
    all_max_km_ts.append(max_kms_ts)
    all_min_km_ts.append(min_kms_ts)


train_data['km_max_distance'] = all_max_km_tr
train_data['km_min_distance'] = all_min_km_tr

test_data['km_max_distance'] = all_max_km_ts
test_data['km_min_distance'] = all_min_km_ts


train_data['h_min__distance']   = sine_distance(train_data,max_lat_tr,max_lon_tr)
train_data['h_min_distance']   = sine_distance(train_data,min_lat_tr,min_lon_tr)

test_data['h_min__distance']   = sine_distance(test_data,max_lat_ts,max_lon_ts)
test_data['h_min_distance']   = sine_distance(test_data,min_lat_ts,min_lon_ts)


train_data['m_max_distance']   = m_sine_distance(train_data,min_lat_tr,min_lon_tr)
train_data['m_min_distance']   = m_sine_distance(train_data,min_lat_tr,min_lon_tr)

test_data['m_max_distance']   = m_sine_distance(test_data,min_lat_ts,min_lon_ts)
test_data['m_min_distance']   = m_sine_distance(test_data,min_lat_ts,min_lon_ts)

train_data['b_max_distance']   = b_sine_distance(train_data,min_lat_tr,min_lon_tr)
train_data['b_min_distance']   = b_sine_distance(train_data,min_lat_tr,min_lon_tr)

test_data['b_max_distance']   = b_sine_distance(test_data,min_lat_ts,min_lon_ts)
test_data['b_min_distance']   = b_sine_distance(test_data,min_lat_ts,min_lon_ts)



k_means_cluster = cluster(train_data[['lat','lon']])
k_means_cluster_ts = cluster(test_data[['lat','lon']])

train_data['Bcounts'] = train_data['B'].map(train_data['B'].value_counts().to_dict())
test_data['Bcounts'] = test_data['B'].map(test_data['B'].value_counts().to_dict())
train_data[['lln_00', 'lln_01']].head(3) #household % with satelite tv


train_data['dx_feat1'] =train_data['stv_01'] - train_data['stv_00']
train_data['dx_feat2'] =train_data['car_00'] - train_data['car_01']
train_data['dx_feat3'] =train_data['lln_00'] - train_data['lln_01']

train_data['dx_stv'] =train_data['stv_01'] * train_data['stv_00']
train_data['dx_car'] =train_data['car_00'] * train_data['car_01']
train_data['dx_lln'] =train_data['lln_00'] * train_data['lln_01']

train_data['dxstv_1'] =train_data['stv_01'] + train_data['stv_00']
train_data['dxcar_2'] =train_data['car_00'] + train_data['car_01']
train_data['dxlln_3'] =train_data['lln_00'] + train_data['lln_01']

test_data['dx_feat1'] =test_data['stv_01'] - test_data['stv_00']
test_data['dx_feat2'] =test_data['car_00'] - test_data['car_01']
test_data['dx_feat3'] =test_data['lln_00'] - test_data['lln_01']

test_data['dx_stv'] =test_data['stv_01'] * test_data['stv_00']
test_data['dx_car'] =test_data['car_00'] *test_data['car_01']
test_data['dx_lln'] =test_data['lln_00'] * test_data['lln_01']

test_data['dxstv_1'] =test_data['stv_01'] + test_data['stv_00']
test_data['dxcar_2'] =test_data['car_00'] +test_data['car_01']
test_data['dxlln_3'] =test_data['lln_00'] + test_data['lln_01']


print(len(test_data.columns))
print(test_data.columns)

train_data['dx/pw_00'] =(train_data['pw_00'] + train_data['pw_01'] + train_data['pw_02'] + train_data['pw_03'] + train_data['pw_04'] + train_data['pw_05'])
test_data['dx/pw_00'] =(test_data['pw_00'] + test_data['pw_01'] + test_data['pw_02'] + test_data['pw_03'] + test_data['pw_04'] + test_data['pw_05'])
train_data['dx/eng_lan'] =train_data['lan_01'] - (train_data['lan_00'] +train_data['lan_02'] + train_data['lan_03']+ train_data['lan_04']+ train_data['lan_05']+ train_data['lan_06']+ train_data['lan_07']+ train_data['lan_08']+ train_data['lan_09']+ train_data['lan_10']++ train_data['lan_11']+ train_data['lan_12']+ train_data['lan_14'])
test_data['dx/eng_lan'] =test_data['lan_01'] - (test_data['lan_00'] +test_data['lan_02'] + test_data['lan_03']+ test_data['lan_04']+ test_data['lan_05'])


len_train= len(train_data)

new_df = pd.concat([train_data, test_data])

train = new_df[:len_train]
train.drop('ADM4_PCODE',axis=1, inplace=True)
train.drop('A',axis=1, inplace=True)
train.drop('B',axis=1, inplace=True)

test = new_df[len_train:]
test.drop("target",axis=1, inplace=True)
test.drop('ADM4_PCODE',axis=1, inplace=True)

test.drop('A',axis=1, inplace=True)
test.drop('B',axis=1, inplace=True)



target= train["target"]
train =train.drop(["ward"], axis=1)
test_id = Samplesub["ward"]
test= test.drop(["ward"], axis=1)



train.drop(['target'], axis=True, inplace=True)
X = train
y = target


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, random_state=1999)
categorical_features_indices = np.where(X.dtypes != np.float)[0]
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
errcb1=[]
y_pred_totcb1=[]
from sklearn.model_selection import KFold,StratifiedKFold, TimeSeriesSplit
fold=KFold(n_splits=31)
i=1

for train_index, test_index in fold.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
   
    m1=CatBoostRegressor(n_estimators=1000,eval_metric='RMSE',max_depth=4,learning_rate=0.21,random_state=33,
                     use_best_model=True)
    m1.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=100,verbose=100)
    preds=m1.predict(X_test)
    print("err: ",sqrt(mean_squared_error(y_test,preds)))
    errcb1.append(sqrt(mean_squared_error(y_test,preds)))
    p1 = m1.predict(test)
    y_pred_totcb1.append(p1)


d = {"ward": test_id, 'target': np.mean(y_pred_totcb1, 0)}
prediction_data = pd.DataFrame(data=d)
prediction_data = prediction_data[["ward", 'target']]

prediction_data.to_csv('woman_sub10.csv', index=False)
