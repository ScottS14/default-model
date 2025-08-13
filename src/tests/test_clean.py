import pandas as pd
from credit_risk import clean as C 
import pytest

# tests for application cleaner
def test_application_sentinel_to_nan(tiny_app):
    out = C.clean_application(tiny_app)
    print(out)
    assert out["DAYS_EMPLOYED"].isna().iloc[0]

def test_application_negative_money_to_nan(tiny_app):
    out = C.clean_application(tiny_app)
    assert pd.isna(out.loc[0, "AMT_INCOME_TOTAL"])

def test_application_constant_drop(tiny_app):
    out = C.clean_application(tiny_app)
    assert "FLAG_CONST" not in out.columns

def test_application_binary_flags(tiny_app):
    out = C.clean_application(tiny_app)
    assert set(out["FLAG_TEST"].dropna().unique()) <= {0, 1}

def test_application_nonnegative_counts_to_nan(tiny_app):
    out = C.clean_application(tiny_app)
    assert pd.isna(out.loc[0, "CNT_CHILDREN"])
    assert pd.isna(out.loc[1, "CNT_FAM_MEMBERS"])

# def test bureau cleaner
def test_bureau_negative_to_nan(tiny_bureau: pd.DataFrame) -> None:
    out = C.clean_bureau(tiny_bureau)
    assert out["AMT_CREDIT_SUM_OVERDUE"].isna().iloc[1]
    assert out["AMT_CREDIT_SUM_OVERDUE"].isna().iloc[2]  


def test_bureau_sentinel_replaced(tiny_bureau: pd.DataFrame) -> None:
    out = C.clean_bureau(tiny_bureau)
    assert out["DAYS_CREDIT"].isna().iloc[0]


def test_bureau_winsorise_upper_cap(tiny_bureau: pd.DataFrame) -> None:
    out = C.clean_bureau(tiny_bureau)
    money_cols = [c for c in out.select_dtypes("number").columns if c.startswith("AMT_")]
    if not money_cols:
        pytest.skip("No AMT_ columns to test winsorisation on")
    caps = out[money_cols].quantile(0.99)
    for col in money_cols:
        tol = abs(caps[col]) * 0.01
        assert out[col].max() <= caps[col] + tol



def test_bureau_no_constant(tiny_bureau: pd.DataFrame) -> None:
    # Create a quasi-constant column to prove it’s removed
    tiny_bureau["ALWAYS_ONE"] = 1
    out = C.clean_bureau(tiny_bureau)
    assert "ALWAYS_ONE" not in out.columns

def test_bureau_nonnegative_counts_to_nan(tiny_bureau):
    out = C.clean_bureau(tiny_bureau)
    assert pd.isna(out.loc[0, "CNT_CREDIT_PROLONG"])
    assert pd.isna(out.loc[0, "CREDIT_DAY_OVERDUE"])
    assert out.loc[2, "CNT_CREDIT_PROLONG"] == 2

# tests for POS cleaner
def test_pos_sentinel_to_nan(tiny_pos):
    out = C.clean_pos(tiny_pos)
    assert out["MONTHS_BALANCE"].isna().iloc[0]

def test_pos_nonnegative_counts_to_nan(tiny_pos):
    out = C.clean_pos(tiny_pos)
    assert pd.isna(out.loc[0, "CNT_INSTALMENT"])
    assert pd.isna(out.loc[0, "CNT_INSTALMENT_FUTURE"])
    assert pd.isna(out.loc[0, "SK_DPD"])
    assert pd.isna(out.loc[0, "SK_DPD_DEF"])
    assert out.loc[1, "CNT_INSTALMENT"] == 5

# tests for Installments cleaner
@pytest.mark.parametrize("col", ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
def test_installments_sentinel_to_nan(tiny_inst, col):
    out = C.clean_installments(tiny_inst)
    assert out[col].isna().iloc[0]

def test_installments_negative_money_to_nan(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert pd.isna(out.loc[1, "AMT_INSTALMENT"])

def test_installments_nonnegative_counts_to_nan(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert pd.isna(out.loc[0, "NUM_INSTALMENT_NUMBER"])
    assert pd.isna(out.loc[0, "NUM_INSTALMENT_VERSION"])
    assert out.loc[1, "NUM_INSTALMENT_NUMBER"] == 4

# tests for Credit-Card cleaner
def test_cc_sentinel_to_nan(tiny_cc):
    out = C.clean_cc(tiny_cc)
    assert out["MONTHS_BALANCE"].isna().iloc[0]

def test_cc_negative_money_to_nan(tiny_cc):
    out = C.clean_cc(tiny_cc)
    assert pd.isna(out.loc[0, "AMT_BALANCE"])

def test_cc_constant_dropped(tiny_cc):
    out = C.clean_cc(tiny_cc)
    assert "NAME_CONTRACT_STATUS" not in out.columns

def test_cc_nonnegative_counts_to_nan(tiny_cc):
    out = C.clean_cc(tiny_cc)
    assert pd.isna(out.loc[0, "SK_DPD"])
    assert pd.isna(out.loc[0, "SK_DPD_DEF"])
    assert out.loc[1, "SK_DPD_DEF"] == 1

# tests for Prev cleaner
def test_prev_negative_money_to_nan(tiny_prev):
    out = C.clean_prev(tiny_prev)
    assert pd.isna(out.loc[0, "AMT_CREDIT"])

def test_prev_binary_flags(tiny_prev):
    out = C.clean_prev(tiny_prev)
    assert set(out["FLAG_TEST"].dropna().unique()) <= {0, 1}
    assert set(out["NFLAG_TEST"].dropna().unique()) <= {0, 1}
