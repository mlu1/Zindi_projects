import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

from lightgbm import LGBMRegressor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

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
to_drop = percentage_missing[percentage_missing>75].keys()

train.drop(to_drop,axis=1,inplace=True)
test.drop(to_drop,axis=1,inplace=True)

print(len(train.columns))
sample_submission = pd.read_csv('data/SampleSubmission.csv')
train_num_df = train.select_dtypes(include=['number'])
print(len(train_num_df.columns))

X = train_num_df.drop('pm2_5', axis = 1)
y = train.pm2_5

test_df = test[X.columns]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

model = LGBMRegressor()
model.fit(X_train, y_train)

# Local score
y_pred = model.predict(X_test)

score = mean_squared_error(y_test, y_pred, squared=False)
print('Local RMSE:', score)
preds = model.predict(test_df)

# Create submission file
sub = pd.DataFrame({'id': test['id'], 'pm2_5': preds})

# Preview sub file
sub.head()
preds = model.predict(test_df)
# Create submission file
sub = pd.DataFrame({'id': test['id'], 'pm2_5': preds})
print(sub.head())
sub.to_csv('submission.csv', index = False)
