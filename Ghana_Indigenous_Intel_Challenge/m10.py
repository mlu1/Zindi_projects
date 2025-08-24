import os
import pandas as pd
import numpy as np


import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm


# Paths (adjust as needed)
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SAMPLE_SUB_PATH = "SampleSubmission.csv"

assert os.path.exists(TRAIN_PATH), f"Missing: {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"Missing: {TEST_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("SampleSubmission columns:", list(sample_sub.columns))
# Inspect columns

cat_cols = ['community', 'district', 'indicator','indicator_description','time_observed']
date_col='prediction_time'

train['Set'] = 'train'
test['Set'] = 'test'
data_all = pd.concat([train, test], axis=0).sort_values(date_col).reset_index(drop=True)


from sklearn import preprocessing
for f in data_all.columns:
    if f in cat_cols:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data_all[f].values) + list(data_all[f].values))
        data_all[f] = lbl.transform(list(data_all[f].values))
    else:
        continue


train = data_all[data_all['Set'] == 'train'].copy()
test = data_all[data_all['Set'] == 'test'].copy()

train.drop(columns=['Set'], inplace=True)
test.drop(columns=['Set'], inplace=True)
     

def parse_time_features(df, time_col='prediction_time'):
    df = df.copy()
    if time_col in df.columns:
        dt = pd.to_datetime(df[time_col].astype(str), dayfirst=True, errors='coerce')
        df['pred_hour'] = dt.dt.hour
        df['pred_dow'] = dt.dt.dayofweek
        df['dow']    = dt.dt.dayofweek
        df['dom']    = dt.dt.dayofyear
        df['quarter']    = dt.dt.quarter
        df['ismonthstart'] = dt.dt.is_month_start
        df['ismonthend'] = dt.dt.is_month_end
        df['dayofweek_name'] = dt.dt.day_name()
        df['is_weekend'] = np.where(df['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
        df['x_month'] = dt.dt.month.astype(int)
        df['x_year'] = dt.dt.year.astype(int) 
        df['sin_dow'] = np.sin(2*np.pi*df['dow']/7)
        df['cos_dow'] = np.cos(2*np.pi*df['dow']/7)
    return df

train = parse_time_features(train)
test  = parse_time_features(test)


def transform(train_df, test_df,date_col='prediction_time'):
        # Sort and tag
    target_col='Target'
    train_df = train_df.sort_values(date_col).reset_index(drop=True)
    test_df = test_df.sort_values(date_col).reset_index(drop=True)
    train_df['Set'] = 'train'
    test_df['Set'] = 'test'
    diff_cols= ['predicted_intensity','confidence','forecast_length']
    dataset = pd.concat([train_df, test_df], axis=0).sort_values(date_col).reset_index(drop=True)

        # Lag and lead features
    for lag in range(1, 2 + 1):
        dataset[f'{target_col}_Lag_{lag}'] = dataset[target_col].shift(lag)
        dataset[f'{target_col}_Lead_{lag}'] = dataset[target_col].shift(-lag)

        # First-order difference features
    for col in diff_cols:
        dataset[f'{col}_diff1'] = dataset[col].diff()


    for step_ in tqdm([1,2,3,4,7,21,30]):
        dataset[f"Target_lag_{step_}"] = dataset["Target"].shift(periods = step_).bfill().bfill()
        for col_ in diff_cols:
            dataset[f"{col_}_lag_{step_}"] = dataset[col_].shift(periods = step_).bfill().bfill()

        # Split back into train and test
    

    TARGET = "Target"
    ID_COL = "ID"

    feature_cols = [c for c in dataset.columns if c not in [TARGET]]
    drop_cols = ["prediction_time", "time_observed", "indicator_description"]
    feature_cols = [c for c in feature_cols if c not in drop_cols]

    numeric_cols = []
    cat_cols_1 = []

    for c in feature_cols:
        if c == ID_COL:
            continue
        if pd.api.types.is_numeric_dtype(dataset[c]):
            numeric_cols.append(c)
        else:
            cat_cols_1.append(c)

    print("Categorical features:", cat_cols_1)
    cat_cols_2 = [x for x in cat_cols_1 if x not in ['Set'] ] 
    print(cat_cols_2)

    for f in list(dataset.columns):
        if f in cat_cols_2:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(dataset[f].values) + list(dataset[f].values))
            dataset[f] = lbl.transform(list(dataset[f].values))
        else:
            continue
    
    print(dataset['Set'].head(10))
    

    train_processed = dataset[dataset['Set'] == 'train'].copy()    
    test_processed = dataset[dataset['Set'] == 'test'].copy()
    
    print(test_processed['Set'].head(10))
 
        # Drop helper column
    train_processed.drop(columns=['Set'], inplace=True)
    test_processed.drop(columns=['Set'], inplace=True)
    
    print(len(train_processed))
    print(len(test_processed))
    
    
    return train_processed, test_processed


train_df,test_df = transform(train,test)


TARGET = "Target"
ID_COL = "ID"

feature_cols = [c for c in train_df.columns if c not in [TARGET]]
drop_cols = ["prediction_time", "time_observed", "indicator_description"]
feature_cols = [c for c in feature_cols if c not in drop_cols]

numeric_cols = []
cat_cols = []

for c in feature_cols:
    if c == ID_COL:
        continue
    if pd.api.types.is_numeric_dtype(train_df[c]):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

print("Numeric features:", numeric_cols)
print("Categorical features:", cat_cols)

unique_categories = train_df[TARGET].unique()
print("Unique values in 'Category' column:")
print(unique_categories)


degree_mapping = {'HEAVYRAIN': 0, 'NORAIN': 1, 'SMALLRAIN': 2, 'MEDIUMRAIN': 3}
train_df[TARGET] = train_df[TARGET].map(degree_mapping)


train_df[TARGET] = train_df[TARGET].astype(int)
y = train_df[[TARGET]]
print(y)
X = train_df.drop(columns=[ID_COL,TARGET,'prediction_time'], axis=1)
test_data = test_df.drop(columns=[ID_COL,TARGET,'prediction_time'], axis=1)

print(X.head(10))

import numpy as np
from catboost import CatBoostClassifier
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression          # ← missing import
from sklearn.model_selection import KFold, cross_val_score

# ---------- Base models -------------------------------------------------
cbc = CatBoostClassifier(
        random_seed=42, verbose=False)

lgbc = LGBMClassifier()
rfc = RandomForestClassifier()

# ---------- Ensembles ---------------------------------------------------

vc = VotingClassifier(
        estimators=[("cbc", cbc),
                    ("lgbc", lgbc),("rfc", rfc)],
         voting="soft"
        )
sc = StackingClassifier(
        estimators=[("cbc", cbc),
                    ("lgbc", lgbc), ("rfc", rfc)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=False, n_jobs=-1)

# ---------- Quick CV helper --------------------------------------------
def cv(model, X, y, name, metric="f1_micro"):
    print(f"→ {name}")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=metric)
    print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}\n")


for model_, model_name_ in zip([cbc,lgbc, rfc, sc, vc][:1], ["catboost","lightgbm", "random forest", "Stacking regressor", "voting regressor"][:1]):
    cv(model_, X, y, model_name_)


for model in [cbc,lgbc,rfc][:-1]:
    print(model)
    model.fit(X, y)


def return_fold_predictions(model):
    ltf = sklearn.preprocessing.FunctionTransformer(func=np.log1p, inverse_func = np.expm1)
    final_preds = list()
    ppm_f1 = list()
    skf = KFold(n_splits = 10, shuffle = bool(1), random_state = 3838)
    for num, (train_idx, val_idx) in enumerate(tqdm(skf.split(X))):
        print(f"Running for split {num+1}")
        split_x_train, split_x_test = X.iloc[train_idx,:], X.iloc[val_idx,:]
        split_y_train, split_y_test = y.iloc[train_idx,:], y.iloc[val_idx,:]
        model.fit(split_x_train, split_y_train)
        print(split_x_test)
        print(model)
        split_pred = model.predict(split_x_test)
        #rmse = np.sqrt(sklearn.metrics.mean_squared_error(split_y_test, split_pred))
        f1s = f1_score(split_y_test, split_pred, average="micro")
        ppm_f1.append(f1s)
        print(f"F1: {f1s}", "\n")
        actual_pred = model.predict(test_data)
        print(actual_pred)
        print(len(actual_pred))
        final_preds.append([i for i in actual_pred])
        
    print(f"RMSE across 10 folds: {np.mean(ppm_f1)}")
    return final_preds


preds = return_fold_predictions(vc)

def conform_to_sample(sample_df: pd.DataFrame, pred_df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """
    Return a DataFrame that has the exact columns and order of sample_df.
    - Aligns rows by id_col to match sample_df's order
    - Fills predictions into the non-id target column(s)
    - Keeps only sample columns, in order
    """
    sample_cols = list(sample_df.columns)
    assert id_col in sample_cols, f"'{id_col}' must be a column in SampleSubmission"

    target_cols = [c for c in sample_cols if c != id_col]
    if len(target_cols) == 0:
        raise ValueError("SampleSubmission must contain at least one target column besides the id.")

    merged = sample_df[[id_col]].merge(pred_df, on=id_col, how="left")

    for tcol in target_cols:
        if tcol in pred_df.columns:
            merged[tcol] = merged[tcol]
        else:
            pred_only = [c for c in pred_df.columns if c != id_col]
            if len(pred_only) == 1:
                merged[tcol] = merged[pred_only[0]]
            else:
                raise ValueError(f"Cannot map predictions to sample target column '{tcol}'. Provide a column named '{tcol}'.")

    return merged[sample_cols]


#X_test = test_df.drop(columns=[ID_COL,'prediction_time'], errors='ignore')


pred_len = len(preds)
print(len(preds[pred_len-1]))
print(len(test))


pred_df = pd.DataFrame({ID_COL: test[ID_COL].values, 'rain_type': preds[pred_len-1]})

#degree_mapping = {'HEAVYRAIN': 0, 'NORAIN': 1, 'SMALLRAIN': 2, 'MEDIUMRAIN': 3}

inv_map = {v: k for k, v in degree_mapping.items()}
pred_df['rain_type'] = pred_df['rain_type'].map(inv_map)

submission = conform_to_sample(sample_sub, pred_df, id_col=ID_COL)

save_path = "submission_baseline.csv"
submission.to_csv(save_path, index=False)
print(f"Saved submission to: {save_path}")
print(submission.head())

# Sanity checks
assert list(submission.columns) == list(sample_sub.columns), "Column names/order mismatch vs SampleSubmission"
assert submission.shape[0] == sample_sub.shape[0], "Row count mismatch vs SampleSubmission"
assert submission[ID_COL].equals(sample_sub[ID_COL]), "ID ordering mismatch vs SampleSubmission"
