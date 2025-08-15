import pandas as pd
import numpy as np
import pytest
from credit_risk import clean as C

S = C.SENTINEL_365243  # 365243 sentinel

@pytest.fixture
def tiny_app() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":            [1, 2],
        "DAYS_EMPLOYED":         [S, -100],   
        "DAYS_LAST_PHONE_CHANGE":[5, -3],     
        "DAYS_ID_PUBLISH":       [0, 12],     
        "DAYS_REGISTRATION":     [-7, 0],    
        "AMT_INCOME_TOTAL":      [0, 200_000],
        "XNA_COL":               ["XNA", "XAP"],
        "FLAG_CONST":            [1, 1],
        "FLAG_TEST":             [1, "NO"],
        "CNT_CHILDREN":          [-1, 2],
        "CNT_FAM_MEMBERS":       [1, -3],
    })

@pytest.fixture
def tiny_bureau() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_BUREAU":           [10, 11, 12],
        "SK_ID_CURR":             [1, 2, 3],
        "AMT_CREDIT_SUM_OVERDUE": [1_000, -50, 0],
        "DAYS_CREDIT":            [S, -200, -400],
        "DAYS_CREDIT_ENDDATE":    [15, -10, 0],   
        "DAYS_ENDDATE_FACT":      [1, -1, np.nan],
        "CREDIT_ACTIVE":          ["Active", "Closed", "Active"],
        "CNT_CREDIT_PROLONG":     [-1, 0, 2],
        "CREDIT_DAY_OVERDUE":     [-5, 0, 10],
    })

@pytest.fixture
def tiny_prev() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":               [1, 2],
        "DAYS_DECISION":            [10, -20],   
        "DAYS_FIRST_DRAWING":       [0, 5],     
        "DAYS_FIRST_DUE":           [-1, 0],    
        "DAYS_LAST_DUE":            [np.nan, 2], 
        "DAYS_TERMINATION":         [-100, 0],
        "AMT_CREDIT":               [-5_000, 250_000],
        "AMT_ANNUITY":              [500, 8_000],
        "NFLAG_TEST":               ['Y', 'NO'],
        "FLAG_TEST":                ['YES', 'FALSE'],
    })

@pytest.fixture
def tiny_pos() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":            [1, 1, 2],
        "MONTHS_BALANCE":        [S, -1, 0],
        "NAME_CONTRACT_STATUS":  ["Unknown"] * 3,
        "CNT_INSTALMENT":        [-1, 5, 2],
        "CNT_INSTALMENT_FUTURE": [-2, 3, 0],
        "SK_DPD":                [-1, 1, 2],
        "SK_DPD_DEF":            [-3, 0, 1],
    })

@pytest.fixture
def tiny_inst() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":         [1, 2],
        "DAYS_INSTALMENT":    [S, -30],   
        "DAYS_ENTRY_PAYMENT": [S, 25],  
        "AMT_PAYMENT":        [0, 1000],
        "AMT_INSTALMENT":     [500, -10],
        "NUM_INSTALMENT_NUMBER":  [-1, 4],
        "NUM_INSTALMENT_VERSION": [-2, 1],
    })

@pytest.fixture
def tiny_cc() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":               [1, 2],
        "MONTHS_BALANCE":           [S, -3],
        "AMT_BALANCE":              [-1, 20_000],
        "AMT_CREDIT_LIMIT_ACTUAL":  [0, 50_000],
        "NAME_CONTRACT_STATUS":     ["Foo", "Foo"],
        "SK_DPD":                   [-4, 0],
        "SK_DPD_DEF":               [-2, 1],
    })
