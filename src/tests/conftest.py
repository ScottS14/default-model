import pandas as pd
import numpy as np
import pytest
from credit_risk import clean as C

S = C.SENTINEL_365243  # the day sentinel constant


# shared tiny fixtures
@pytest.fixture
def tiny_app() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":       [1, 2],
        "DAYS_EMPLOYED":    [S, 100],
        "AMT_INCOME_TOTAL": [0, 200_000],
        "XNA_COL":          ["XNA", "XAP"],
        "FLAG_CONST":       [1, 1],
    })

@pytest.fixture
def tiny_bureau() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_BUREAU":           [10, 11, 12],
        "SK_ID_CURR":             [1, 2, 3],
        "AMT_CREDIT_SUM_OVERDUE": [1_000, -50, 0],
        "DAYS_CREDIT":            [S, -200, -400],
        "CREDIT_ACTIVE":          ["Active", "Closed", "Active"],  
    })

@pytest.fixture
def tiny_prev() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR":  [1, 2],
            "AMT_CREDIT":  [-5_000, 250_000],  
            "AMT_ANNUITY": [   500,   8_000],
        }
    )

@pytest.fixture
def tiny_pos() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":          [1, 1, 2],
        "MONTHS_BALANCE":      [S, -1, 0],
        "NAME_CONTRACT_STATUS":["Unknown"] * 3,
        "SK_DPD":              [0, 1, 2],
    })

@pytest.fixture
def tiny_inst() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":       [1, 2],
        "DAYS_INSTALMENT":  [S, -30],
        "DAYS_ENTRY_PAYMENT":[S, -25],
        "AMT_PAYMENT":      [0, 1000],
        "AMT_INSTALMENT":   [500, -10],
    })

@pytest.fixture
def tiny_cc() -> pd.DataFrame:
    return pd.DataFrame({
        "SK_ID_CURR":             [1, 2],
        "MONTHS_BALANCE":         [S, -3],
        "AMT_BALANCE":            [-1, 20_000],
        "AMT_CREDIT_LIMIT_ACTUAL":[0, 50_000],
        "NAME_CONTRACT_STATUS":   ["Foo", "Foo"],
    })


