import pandas as pd
from credit_risk import clean as C
import pytest

# Helper: some day-columns can end up fully-NaN and be dropped by the cleaner.
def assert_all_nan_or_absent(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        assert df[col].isna().all()
    else:
        # fully-NaN and dropped is acceptable
        assert True


# ----------------- application -----------------

def test_application_sentinel_to_nan(tiny_app):
    out = C.clean_application(tiny_app)
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

def test_application_nonpositive_days_enforced(tiny_app):
    out = C.clean_application(tiny_app)
    assert pd.isna(out.loc[0, "DAYS_LAST_PHONE_CHANGE"])
    assert pd.isna(out.loc[1, "DAYS_ID_PUBLISH"])
    assert out.loc[1, "DAYS_REGISTRATION"] == 0
    assert out.loc[1, "DAYS_EMPLOYED"] == -100


# ----------------- bureau -----------------

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

def test_bureau_nonpositive_days_enforced(tiny_bureau):
    out = C.clean_bureau(tiny_bureau)
    assert pd.isna(out.loc[0, "DAYS_CREDIT_ENDDATE"])
    assert pd.isna(out.loc[0, "DAYS_ENDDATE_FACT"])
    assert out.loc[1, "DAYS_CREDIT_ENDDATE"] == -10
    assert out.loc[2, "DAYS_CREDIT_ENDDATE"] == 0

def test_bureau_no_constant(tiny_bureau: pd.DataFrame) -> None:
    tiny_bureau["ALWAYS_ONE"] = 1
    out = C.clean_bureau(tiny_bureau)
    assert "ALWAYS_ONE" not in out.columns

def test_bureau_nonnegative_counts_to_nan(tiny_bureau):
    out = C.clean_bureau(tiny_bureau)
    assert pd.isna(out.loc[0, "CNT_CREDIT_PROLONG"])
    assert pd.isna(out.loc[0, "CREDIT_DAY_OVERDUE"])
    assert out.loc[2, "CNT_CREDIT_PROLONG"] == 2


# ----------------- POS -----------------

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


# ----------------- installments -----------------

def test_installments_sentinel_to_nan_days_instalment(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert out["DAYS_INSTALMENT"].isna().iloc[0]
    assert out.loc[1, "DAYS_INSTALMENT"] == -30

def test_installments_entry_payment_may_be_dropped(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert_all_nan_or_absent(out, "DAYS_ENTRY_PAYMENT")

def test_installments_negative_money_to_nan(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert pd.isna(out.loc[1, "AMT_INSTALMENT"])

def test_installments_nonnegative_counts_to_nan(tiny_inst):
    out = C.clean_installments(tiny_inst)
    assert pd.isna(out.loc[0, "NUM_INSTALMENT_NUMBER"])
    assert pd.isna(out.loc[0, "NUM_INSTALMENT_VERSION"])
    assert out.loc[1, "NUM_INSTALMENT_NUMBER"] == 4


# ----------------- credit card -----------------

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


# ----------------- previous_application -----------------

def test_prev_negative_money_to_nan(tiny_prev):
    out = C.clean_prev(tiny_prev)
    assert pd.isna(out.loc[0, "AMT_CREDIT"])

def test_prev_binary_flags(tiny_prev):
    out = C.clean_prev(tiny_prev)
    assert set(out["FLAG_TEST"].dropna().unique()) <= {0, 1}
    assert set(out["NFLAG_TEST"].dropna().unique()) <= {0, 1}

def test_prev_nonpositive_days_enforced(tiny_prev):
    out = C.clean_prev(tiny_prev)
    assert pd.isna(out.loc[0, "DAYS_DECISION"])
    assert out.loc[1, "DAYS_DECISION"] == -20
    assert out.loc[0, "DAYS_FIRST_DRAWING"] == 0
    assert pd.isna(out.loc[1, "DAYS_FIRST_DRAWING"])
    assert_all_nan_or_absent(out, "DAYS_LAST_DUE")
    for c in [c for c in out.columns if c.startswith("DAYS_")]:
        assert (out[c].dropna() <= 0).all()
