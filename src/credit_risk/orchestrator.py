import pandas as pd

from credit_risk.clean import clean_application, clean_bureau
from credit_risk.feat_eng import compute_app_ratios, aggregate_bureau

app = pd.read_csv('data/raw/home_credit/application_train.csv')
bureau = pd.read_csv('data/raw/home_credit/bureau.csv')
bureau_balance = pd.read_csv('data/raw/home_credit/bureau_balance.csv')

clean_app = clean_application(app)
clean_b = clean_bureau(bureau)

agg_app = compute_app_ratios(clean_app)
agg_b = aggregate_bureau(bureau, bureau_balance)

print(agg_app.head())
print(agg_b.head())