import pandas as pd
from datetime import datetime
import numpy as np
import pytest

from recurrent_health_events_prediction.preprocessing.utils import calculate_past_rolling_stats, get_rows_up_to_event_id

def test_get_rows_up_to_event_id():
    # Sample data
    data = {
        'SUBJECT_ID': ['A', 'A', 'A', 'B', 'B', 'C'],
        'EVENT_ID': [101, 102, 103, 201, 202, 301],
        'ADMITTIME': [
            datetime(2021, 1, 1),
            datetime(2021, 2, 1),
            datetime(2021, 3, 1),
            datetime(2021, 1, 15),
            datetime(2021, 2, 15),
            datetime(2021, 3, 10)
        ],
        'VALUE': [10, 20, 30, 5, 15, 99],
    }
    df = pd.DataFrame(data)

    # Event IDs to truncate to (per subject)
    event_ids = pd.Series({
        'A': 102,
        'B': 202,
        'C': 999  # nonexistent, should exclude subject C
    })

    result = get_rows_up_to_event_id(
        df=df,
        event_id_col='EVENT_ID',
        event_ids=event_ids,
        id_col='SUBJECT_ID',
        time_col='ADMITTIME',
        include_event_id=True
    )

    # Expected output: rows up to EVENT_ID 102 for A, and 202 for B only
    expected_event_ids = [101, 102, 201, 202]
    assert list(result['EVENT_ID']) == expected_event_ids
    assert 'C' not in result['SUBJECT_ID'].values
    
    # For the case where include_event_id is False
    result_no_event = get_rows_up_to_event_id(
        df=df,
        event_id_col='EVENT_ID',
        event_ids=event_ids,
        id_col='SUBJECT_ID',
        time_col='ADMITTIME',
        include_event_id=False
    )
    
    # Expected output: rows up to EVENT_ID 102 for A, and 201 for B, excluding the event_id row
    expected_event_ids_no_event = [101, 201]
    
    assert list(result_no_event['EVENT_ID']) == expected_event_ids_no_event
    assert 'C' not in result_no_event['SUBJECT_ID'].values

@pytest.fixture
def sample_df():
    # 2 patients, 4 visits each
    return pd.DataFrame({
        'SUBJECT_ID': [1, 1, 1, 1, 2, 2, 2, 2],
        'DAYS_UNTIL_NEXT_HOSPITALIZATION': [45, 31, 20, np.nan, 45, 24, 22, np.nan],
        'READMISSION_30_DAYS': [0, 0, 1, np.nan, 0, 1, 1, np.nan],
    })

def test_calculate_past_rolling_stats_mean_sum(sample_df):
    # Test mean and sum on binary variable
    stats = calculate_past_rolling_stats(
        sample_df, group_col='SUBJECT_ID', feature='READMISSION_30_DAYS',
        stats=['mean', 'sum'], prefix='READM_30_DAYS'
    )

    # For patient 1: 
    #   READMISSION_30_DAYS: [0, 0, 1, np.na]
    #   shifted READMISSION_30_DAYS: [pd.na, 0, 0, 1]
    # means (past): [nan, 0, 0, (0+0+1)/3] = [nan, 0, 0, 0.333]
    # count: [0, 0, 0, 1]
    # For patient 2:
    #   READMISSION_30_DAYS: [0, 1, 1, np.na]
    #   shifted READMISSION_30_DAYS: [pd.na, 0, 1, 1]
    # means (past): [nan, 0, (0+1)/2, (0+1+1)/3] = [nan, 0, 0.5, 0.666]
    # count: [0, 0, 1, 2]
    
    expected_mean = [np.nan, 0.0, 0.0, 0.33333333, np.nan, 0.0, 0.500, 0.66666666]
    expected_count = [np.nan, 0, 0, 1, np.nan, 0, 1, 2]

    np.testing.assert_array_almost_equal(stats['READM_30_DAYS_PAST_MEAN'].values, expected_mean, err_msg="Mean failed")
    np.testing.assert_array_almost_equal(stats['READM_30_DAYS_PAST_SUM'].values, expected_count, err_msg="Count failed")

def test_calculate_past_rolling_stats_median(sample_df):
    stats = calculate_past_rolling_stats(
        sample_df, group_col='SUBJECT_ID', feature='DAYS_UNTIL_NEXT_HOSPITALIZATION',
        stats=['median'], prefix='DAYS_NEXT_HOSP'
    )

    # For patient 1:
    #   DAYS_UNTIL_NEXT_HOSPITALIZATION: [45, 31, 20, nan]
    #   Shifted: [nan, 45, 31, 20]
    #   Rolling medians:
    #     1st: nan            (no previous data)
    #     2nd: median([45])          = 45.0
    #     3rd: median([45, 31])      = 38.0
    #     4th: median([45, 31, 20])  = 31.0

    # For patient 2:
    #   DAYS_UNTIL_NEXT_HOSPITALIZATION: [45, 24, 22, nan]
    #   Shifted: [nan, 45, 24, 22]
    #   Rolling medians:
    #     1st: nan                 (no previous data)
    #     2nd: median([45])        = 45.0
    #     3rd: median([45, 24])    = 34.5
    #     4th: median([45, 24, 22])= 24.0

    expected_median = [np.nan, 45.0, 38.0, 31.0, np.nan, 45.0, 34.5, 24.0]

    np.testing.assert_array_almost_equal(
        stats['DAYS_NEXT_HOSP_PAST_MEDIAN'].values, expected_median, decimal=3,
        err_msg="Median failed"
    )


if __name__ == "__main__":
    pytest.main([__file__])
