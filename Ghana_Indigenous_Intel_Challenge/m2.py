import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import feature_engine
from feature_engine.creation import CyclicalFeatures
from feature_engine.datetime import DatetimeFeatures
from feature_engine.transformation import YeoJohnsonTransformer, PowerTransformer, LogCpTransformer, BoxCoxTransformer
from feature_engine.selection import DropConstantFeatures,DropCorrelatedFeatures, DropDuplicateFeatures
from feature_engine.outliers import OutlierTrimmer



import matplotlib.pyplot as plt

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


def parse_time_features(df, time_col='prediction_time'):
    df = df.copy()
    if time_col in df.columns:
        dt = pd.to_datetime(df[time_col].astype(str), dayfirst=True, errors='coerce')
        df['pred_hour'] = dt.dt.hour
        df['pred_dow'] = dt.dt.dayofweek
        df['pred_date'] = dt.dt.date.astype('str')
        #df['sin_hour'] = dt.sin(2*np.pi*df['pred_hour']/24)
        #df['cos_hour'] = dt.cos(2*np.pi*df['pred_hour']/24)
        df['dow']    = dt.dt.dayofweek
        df['sin_dow'] = np.sin(2*np.pi*df['dow']/7)
        df['cos_dow'] = np.cos(2*np.pi*df['dow']/7)
    return df




def engineer_features(df_:pd.DataFrame) -> pd.DataFrame:
    #df_["Target"] = df_["Target"].fillna(value = -999)
    #df_["Target"] = df_["Target"].bfill().bfill()
    #for step_ in tqdm([1,2,3,4,7,21,30]):
    #    df_[f"Target_lag_{step_}"] = df_["Target"].shift(periods = step_).bfill().bfill()
    #    for col_ in ["confidence", "predicted_intensity"]:
    #        df_[f"{col_}_lag_{step_}"] = df_[col_].shift(periods = step_).bfill().bfill()
    dcf = DropConstantFeatures(
        tol = 1
    )
    dccf = DropCorrelatedFeatures(
        threshold = .8
    )
    
    ddf = DropDuplicateFeatures(
        missing_values = "ignore"
    )
    
    dtf = DatetimeFeatures(
        variables = ["pred_date"],
        drop_original = True,
        features_to_extract = ["month", "quarter", "year", "week", "day_of_month", "day_of_year"]
    )
    gcf = CyclicalFeatures(
        variables = ["pred_date" + "_" + i for i in ["month", "day_of_year", "day_of_month"]]
    )

    #ddf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #dtf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #dcf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #gcf.replace([np.inf, -np.inf], np.nan, inplace=True)
    #dccf.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    df_ = dtf.fit_transform(df_)
    print(df_)
    #print(fuck)


    df_ = ddf.fit_transform(df_)
    #df_ = dtf.fit_transform(df_)
    #df_ = dcf.fit_transform(df_)
    df_ = gcf.fit_transform(df_)
    #df_ = dccf.fit_transform(df_)
    #print("Constant Features: ", dcf.features_to_drop_)
    #print("Correlated feature Pairs: ", dccf.correlated_feature_sets_)
    #print("Duplicate Features: ", ddf.features_to_drop_)
    
    return df_





train = parse_time_features(train)
test  = parse_time_features(test)



train = engineer_features(train)
test  = engineer_features(test)




TARGET = "Target"
ID_COL = "ID"

feature_cols = [c for c in train.columns if c not in [TARGET]]
drop_cols = ["prediction_time", "time_observed", "indicator_description"]
feature_cols = [c for c in feature_cols if c not in drop_cols]

numeric_cols = []
cat_cols = []

for c in feature_cols:
    if c == ID_COL:
        continue
    if pd.api.types.is_numeric_dtype(train[c]):
        numeric_cols.append(c)
    else:
        cat_cols.append(c)

print("Numeric features:", numeric_cols)
print("Categorical features:", cat_cols)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)





'''
model = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42
)
model = RandomForestClassifier()
'''

import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',    # macro-F1 cares about minorities
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)

pipe = Pipeline([
    ('pre', preprocess),        # same ColumnTransformer as before
    ('model', lgb_clf)
])

'''
pipe = Pipeline(steps=[('preprocess', preprocess),
                      ('model', model)])
'''



X = train[feature_cols].drop(columns=[ID_COL], errors='ignore')
y = train[TARGET]


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = cross_val_score(pipe, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"CV Macro F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"CV Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")


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

pipe.fit(X, y)

X_test = test[feature_cols].drop(columns=[ID_COL], errors='ignore')
test_pred = pipe.predict(X_test)

pred_df = pd.DataFrame({ID_COL: test[ID_COL].values, 'rain_type': test_pred})

submission = conform_to_sample(sample_sub, pred_df, id_col=ID_COL)

save_path = "submission_baseline.csv"
submission.to_csv(save_path, index=False)
print(f"Saved submission to: {save_path}")
print(submission.head())

# Sanity checks
assert list(submission.columns) == list(sample_sub.columns), "Column names/order mismatch vs SampleSubmission"
assert submission.shape[0] == sample_sub.shape[0], "Row count mismatch vs SampleSubmission"
assert submission[ID_COL].equals(sample_sub[ID_COL]), "ID ordering mismatch vs SampleSubmission"



