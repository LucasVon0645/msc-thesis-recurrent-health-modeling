import pytest
import pandas as pd
import numpy as np
from recurrent_health_events_prediction.preprocessing.feature_extraction import FeatureExtractorDrugRelapse

@pytest.fixture
def donor_positives_fixture():
    data = [
        # Donor A: classic run, resets, consecutive zeros
        {"DONOR_ID": "A", "DRUG_POSITIVE": 1, "TIME": "2024-01-01"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 1, "TIME": "2024-01-02"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 0, "TIME": "2024-01-03"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 1, "TIME": "2024-01-04"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 1, "TIME": "2024-01-05"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 1, "TIME": "2024-01-06"},
        {"DONOR_ID": "A", "DRUG_POSITIVE": 0, "TIME": "2024-01-07"},

        # Donor B: all negatives
        {"DONOR_ID": "B", "DRUG_POSITIVE": 0, "TIME": "2024-01-01"},
        {"DONOR_ID": "B", "DRUG_POSITIVE": 0, "TIME": "2024-01-02"},
        {"DONOR_ID": "B", "DRUG_POSITIVE": 0, "TIME": "2024-01-03"},

        # Donor C: all positives
        {"DONOR_ID": "C", "DRUG_POSITIVE": 1, "TIME": "2024-01-01"},
        {"DONOR_ID": "C", "DRUG_POSITIVE": 1, "TIME": "2024-01-02"},
        {"DONOR_ID": "C", "DRUG_POSITIVE": 1, "TIME": "2024-01-03"},

        # Donor D: positives, then 0s, then another positive
        {"DONOR_ID": "D", "DRUG_POSITIVE": 1, "TIME": "2024-01-01"},
        {"DONOR_ID": "D", "DRUG_POSITIVE": 1, "TIME": "2024-01-02"},
        {"DONOR_ID": "D", "DRUG_POSITIVE": 0, "TIME": "2024-01-03"},
        {"DONOR_ID": "D", "DRUG_POSITIVE": 0, "TIME": "2024-01-04"},
        {"DONOR_ID": "D", "DRUG_POSITIVE": 1, "TIME": "2024-01-05"},
    ]
    df = pd.DataFrame(data)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df

def test_num_positives_since_last_negative(donor_positives_fixture):
    out = FeatureExtractorDrugRelapse._get_number_positives_since_last_negative(donor_positives_fixture)

    # Donor A expected: [NaN, NaN, 2, NaN, NaN, Nan, 3]
    expected_A = [np.nan, np.nan, 2, np.nan, np.nan, np.nan, 3.0]
    a_out = out[out["DONOR_ID"] == "A"]["NUM_POSITIVES_SINCE_LAST_NEGATIVE"].tolist()
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(a_out, expected_A))

    # Donor B expected: [0, 0, 0]
    expected_B = [0, 0, 0]
    b_out = out[out["DONOR_ID"] == "B"]["NUM_POSITIVES_SINCE_LAST_NEGATIVE"].tolist()
    assert b_out == expected_B

    # Donor C expected: [NaN, NaN, NaN]
    expected_C = [np.nan, np.nan, np.nan]
    c_out = out[out["DONOR_ID"] == "C"]["NUM_POSITIVES_SINCE_LAST_NEGATIVE"].tolist()
    assert all(pd.isna(x) for x in c_out)

    # Donor D expected: [NaN, NaN, 2, 0, NaN]
    expected_D = [np.nan, np.nan, 2, 0, np.nan]
    d_out = out[out["DONOR_ID"] == "D"]["NUM_POSITIVES_SINCE_LAST_NEGATIVE"].tolist()
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(d_out, expected_D))

def test_num_negatives_since_last_positive(donor_positives_fixture):
    out = FeatureExtractorDrugRelapse._get_num_negatives_since_last_positive(donor_positives_fixture)

    # Donor A: [NaN, 0, 0, 1, 0, 0, 0]
    expected_A = [np.nan, 0, 0, 1, 0, 0, 0]
    a_out = out[out["DONOR_ID"] == "A"]["NUM_NEGATIVES_SINCE_LAST_POSITIVE"].tolist()
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(a_out, expected_A))

    # Donor B: all 0s, no 1s, should be all NaN
    b_out = out[out["DONOR_ID"] == "B"]["NUM_NEGATIVES_SINCE_LAST_POSITIVE"].tolist()
    assert all(pd.isna(x) for x in b_out)

    # Donor C: all 1s, should be first NaN and rest 0s
    expected_C = [np.nan, 0, 0]
    c_out = out[out["DONOR_ID"] == "C"]["NUM_NEGATIVES_SINCE_LAST_POSITIVE"].tolist()
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(c_out, expected_C))

    # Donor D: [NaN, 0, 0, 1, 2]
    expected_D = [np.nan, 0, 0, 1, 2]
    d_out = out[out["DONOR_ID"] == "D"]["NUM_NEGATIVES_SINCE_LAST_POSITIVE"].tolist()
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(d_out, expected_D))

def test_get_time_since_last_negative(donor_positives_fixture):
    """Test the _get_time_since_last_negative function."""
    result_df = FeatureExtractorDrugRelapse._get_time_since_last_negative(donor_positives_fixture)

    # Donor A
    result_donor_a_df = result_df[result_df["DONOR_ID"] == "A"]
    expected_times = [np.nan, np.nan, np.nan, 1, 2, 3, 4]
    # Assert the results match the expected values
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(result_donor_a_df["TIME_SINCE_LAST_NEGATIVE"].tolist(), expected_times))

    # Donor B
    result_donor_b_df = result_df[result_df["DONOR_ID"] == "B"]
    expected_times_b = [np.nan, 1, 1]
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(result_donor_b_df["TIME_SINCE_LAST_NEGATIVE"].tolist(), expected_times_b))

    # Donor C
    result_donor_c_df = result_df[result_df["DONOR_ID"] == "C"]
    assert all(pd.isna(x) for x in result_donor_c_df["TIME_SINCE_LAST_NEGATIVE"].tolist())

    # Donor D
    result_donor_d_df = result_df[result_df["DONOR_ID"] == "D"]
    expected_times_d = [np.nan, np.nan, np.nan, 1, 1]
    assert all((pd.isna(a) and pd.isna(b)) or (a == b) for a, b in zip(result_donor_d_df["TIME_SINCE_LAST_NEGATIVE"].tolist(), expected_times_d))