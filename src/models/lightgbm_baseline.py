import lightgbm as lgb
import pandas as pd

# creating lightgbm dataset 
load_data = pd.read_parquet('data/processed/train_features_lgbm.parquet')

cols = [col for col in load_data.columns if col != 'TARGET']

X = load_data[cols]
y = load_data['TARGET']

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,     
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "is_unbalance": True,      
    "seed": 42                 
}

dtrain = lgb.Dataset(X, label=y)

cv = lgb.cv(
    params,
    dtrain,
    num_boost_round=5000,         
    nfold=5,
    stratified=True,
    shuffle=True,                 
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

print(cv)

