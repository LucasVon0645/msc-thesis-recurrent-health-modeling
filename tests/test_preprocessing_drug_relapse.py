import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from recurrent_health_events_prediction.preprocessing.preprocessors import DataPreprocessorDrugRelapse

@pytest.fixture
def example_drug_tests_df():
    # Build cases for Donor A, B, C
    data = [
        # Donor A: P-N-N-P-N (relapse after first positive, the second relapse should be ignored)
        {"DONOR_ID": "A", "COLLECTION_ID": 1, "TIME": "2024-01-01", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "A", "COLLECTION_ID": 2, "TIME": "2024-01-03", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "A", "COLLECTION_ID": 3, "TIME": "2024-01-05", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "A", "COLLECTION_ID": 4, "TIME": "2024-01-10", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "A", "COLLECTION_ID": 5, "TIME": "2024-01-15", "DRUG_POSITIVE": 0},


        # Donor B: N-N-N (should be relapse-free, censored)
        {"DONOR_ID": "B", "COLLECTION_ID": 6, "TIME": "2024-02-01", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "B", "COLLECTION_ID": 7, "TIME": "2024-02-03", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "B", "COLLECTION_ID": 8, "TIME": "2024-02-04", "DRUG_POSITIVE": 0},

        # Donor C: P-P-N-N (should start relapse after second positive)
        {"DONOR_ID": "C", "COLLECTION_ID": 9, "TIME": "2024-03-01", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "C", "COLLECTION_ID": 10, "TIME": "2024-03-03", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "C", "COLLECTION_ID": 11, "TIME": "2024-03-07", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "C", "COLLECTION_ID": 12, "TIME": "2024-03-10", "DRUG_POSITIVE": 0},

        # Donor D: N-P-N-N-P-N-N (should start relapse after first positive)
        {"DONOR_ID": "D", "COLLECTION_ID": 13, "TIME": "2024-04-01", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "D", "COLLECTION_ID": 14, "TIME": "2024-05-02", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "D", "COLLECTION_ID": 15, "TIME": "2024-06-05", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "D", "COLLECTION_ID": 16, "TIME": "2024-06-10", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "D", "COLLECTION_ID": 90, "TIME": "2024-06-12", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "D", "COLLECTION_ID": 80, "TIME": "2024-07-02", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "D", "COLLECTION_ID": 81, "TIME": "2024-07-05", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "D", "COLLECTION_ID": 82, "TIME": "2024-07-10", "DRUG_POSITIVE": 0},


        # Donor F: P-N-N-P (explicit duration test)
        {"DONOR_ID": "F", "COLLECTION_ID": 17, "TIME": "2024-06-01", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "F", "COLLECTION_ID": 18, "TIME": "2024-06-04", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "F", "COLLECTION_ID": 19, "TIME": "2024-06-07", "DRUG_POSITIVE": 0},
        {"DONOR_ID": "F", "COLLECTION_ID": 20, "TIME": "2024-06-10", "DRUG_POSITIVE": 1},
    ]
    df = pd.DataFrame(data)
    df["TIME"] = pd.to_datetime(df["TIME"])
    return df

def test_extract_relapse_periods_shapes_and_columns(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    expected_columns = [
        "DONOR_ID", "COLLECTION_ID", "PREV_POS_COLLECTION_ID",
        "RELAPSE_START", "RELAPSE_END",
        "EVENT_DURATION", "RELAPSE_EVENT", "NUM_TESTS_PERIOD"
    ]
    for col in expected_columns:
        assert col in result.columns

def test_donor_A_relapse_periods(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    donor_a = result[result["DONOR_ID"] == "A"].reset_index(drop=True)

    # Should be two periods: one after first positive (relapse), one after second positive (censored)
    assert len(donor_a) == 1

    # First period: relapse event (after 2024-01-01, first negative at 2024-01-03, ends with next positive at 2024-01-10)
    assert donor_a.loc[0, "COLLECTION_ID"] == 2
    assert donor_a.loc[0, "PREV_POS_COLLECTION_ID"] == 1
    assert donor_a.loc[0, "RELAPSE_EVENT"] == 1
    assert donor_a.loc[0, "RELAPSE_START"] == pd.Timestamp("2024-01-03")
    assert donor_a.loc[0, "RELAPSE_END"] == pd.Timestamp("2024-01-10")
    assert donor_a.loc[0, "NUM_TESTS_PERIOD"] == 2
    assert donor_a.loc[0, "EVENT_DURATION"] == (pd.Timestamp("2024-01-10") - pd.Timestamp("2024-01-03")).days

def test_donor_B_relapse_free_period(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    donor_b = result[result["DONOR_ID"] == "B"].reset_index(drop=True)

    # Should be only one period, relapse-free
    assert len(donor_b) == 1
    assert donor_b.loc[0, "COLLECTION_ID"] == 6
    assert pd.isna(donor_b.loc[0, "PREV_POS_COLLECTION_ID"])
    assert donor_b.loc[0, "RELAPSE_EVENT"] == 0
    assert donor_b.loc[0, "NUM_TESTS_PERIOD"] == 3
    assert donor_b.loc[0, "EVENT_DURATION"] == (pd.Timestamp("2024-02-04") - pd.Timestamp("2024-02-01")).days

def test_relapse_start_end_equal(example_drug_tests_df):
    # Artificially make a case where end==start
    # In this case, the relapse period should not be created
    df = example_drug_tests_df.copy()
    # Donor X: P-N with same day
    new_rows = pd.DataFrame([
        {"DONOR_ID": "X", "COLLECTION_ID": 99, "TIME": "2024-05-01", "DRUG_POSITIVE": 1},
        {"DONOR_ID": "X", "COLLECTION_ID": 99, "TIME": "2024-05-01", "DRUG_POSITIVE": 0},
    ])
    df = pd.concat([df, new_rows], ignore_index=True)
    df["TIME"] = pd.to_datetime(df["TIME"])

    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(df)
    donor_x = result[result["DONOR_ID"] == "X"].reset_index(drop=True)
    
    assert len(donor_x) == 0

def test_donor_C_relapse_periods(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    donor_c = result[result["DONOR_ID"] == "C"].reset_index(drop=True)

    # Should be one period: after second positive (2024-03-01), ends with second negative (2024-03-10)
    assert len(donor_c) == 1
    assert donor_c.loc[0, "COLLECTION_ID"] == 11
    assert donor_c.loc[0, "PREV_POS_COLLECTION_ID"] == 10
    assert donor_c.loc[0, "RELAPSE_EVENT"] == 0
    assert donor_c.loc[0, "RELAPSE_START"] == pd.Timestamp("2024-03-07")
    assert donor_c.loc[0, "RELAPSE_END"] == pd.Timestamp("2024-03-10")
    assert donor_c.loc[0, "NUM_TESTS_PERIOD"] == 2
    assert donor_c.loc[0, "EVENT_DURATION"] == (pd.Timestamp("2024-03-10") - pd.Timestamp("2024-03-07")).days

def test_donor_D_relapse_periods(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    donor_d = result[result["DONOR_ID"] == "D"].reset_index(drop=True)

    # Should be two periods:
    # - first period: 2024-06-05 to 2024-07-02 (3 zeros in the sequence)
    # - second period: 2024-07-05 to 2024-07-10 (2 zeros in the sequence)
    # - obs: the first zero is ignored, because we dont know the start of the relapse
    assert len(donor_d) == 2

    assert donor_d.loc[0, "RELAPSE_EVENT"] == 1
    assert donor_d.loc[0, "RELAPSE_START"] == pd.Timestamp("2024-06-05")
    assert donor_d.loc[0, "RELAPSE_END"] == pd.Timestamp("2024-07-02")
    assert donor_d.loc[0, "NUM_TESTS_PERIOD"] == 3
    assert donor_d.loc[0, "EVENT_DURATION"] == (pd.Timestamp("2024-07-02") - pd.Timestamp("2024-06-05")).days
    assert donor_d.loc[0, "COLLECTION_ID"] == 15
    assert donor_d.loc[0, "PREV_POS_COLLECTION_ID"] == 14

    assert donor_d.loc[1, "RELAPSE_EVENT"] == 0
    assert donor_d.loc[1, "RELAPSE_START"] == pd.Timestamp("2024-07-05")
    assert donor_d.loc[1, "RELAPSE_END"] == pd.Timestamp("2024-07-10")
    # The second period has 2 tests
    assert donor_d.loc[1, "NUM_TESTS_PERIOD"] == 2
    assert donor_d.loc[1, "EVENT_DURATION"] == (pd.Timestamp("2024-07-10") - pd.Timestamp("2024-07-05")).days
    assert donor_d.loc[1, "COLLECTION_ID"] == 81
    assert donor_d.loc[1, "PREV_POS_COLLECTION_ID"] == 80

def test_donor_F_relapse_periods(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    donor_f = result[result["DONOR_ID"] == "F"].reset_index(drop=True)

    # Should be one period: after first positive (2024-06-01), ends with second positive (2024-06-10)
    assert len(donor_f) == 1
    assert donor_f.loc[0, "RELAPSE_EVENT"] == 1
    assert donor_f.loc[0, "RELAPSE_START"] == pd.Timestamp("2024-06-04")
    assert donor_f.loc[0, "RELAPSE_END"] == pd.Timestamp("2024-06-10")
    assert donor_f.loc[0, "NUM_TESTS_PERIOD"] == 2
    assert donor_f.loc[0, "EVENT_DURATION"] == (pd.Timestamp("2024-06-10") - pd.Timestamp("2024-06-04")).days
    assert donor_f.loc[0, "COLLECTION_ID"] == 18
    assert donor_f.loc[0, "PREV_POS_COLLECTION_ID"] == 17

def test_handles_multiple_donors(example_drug_tests_df):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._extract_relapse_periods(example_drug_tests_df)
    # Should have periods for all donors present
    donors = set(example_drug_tests_df["DONOR_ID"])
    assert donors.issuperset(set(result["DONOR_ID"]))

@pytest.fixture
def relapses_df_fixture():
    # DONOR 1: 2 relapses, last one should be flagged, because NUM_PREV_RELAPSES=1 for it;
    # DONOR 2: only censored events, last one should be flagged
    # DONOR 3: 2 events,first one has NUM_PREV_RELAPSES=0, so last event must be taken;
    data = {
        'DONOR_ID':     [1, 1, 1, 2, 2, 3, 3],
        'RELAPSE_START': [10, 20, 30, 15, 40, 12, 20],
        'NUM_PREV_RELAPSES': [0, 1, 2, 0, 1, 0, 1],
        'COLLECTION_ID':[101, 102, 103, 201, 202, 203, 204],
        'RELAPSE_EVENT':[1,   0,   1,   0,   0, 1, 0],
    }
    return pd.DataFrame(data)

def test_define_last_events(relapses_df_fixture):
    preprocessor = DataPreprocessorDrugRelapse({})
    result = preprocessor._define_last_events(relapses_df_fixture.copy())

    # Expected IS_LAST_EVENT:
    # DONOR 1: rows [0,1,2], last RELAPSE_EVENT=1 at index 2 (COLLECTION_ID=103) -> flagged
    # DONOR 2: rows [3,4], last RELAPSE_EVENT=0 at index 4 (COLLECTION_ID=202) -> flagged

    expected = [0, 0, 1, 0, 1, 0, 1]
    assert result['IS_LAST_EVENT'].tolist() == expected

@pytest.fixture
def sample_drug_tests_and_relapses_for_historical_split():
    relapses_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 2],
        'RELAPSE_START': [100, 200, 300],
        'IS_LAST_EVENT': [0, 1, 1]
    })
    drug_tests_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2],
        'TIME': [50, 120, 220, 250, 350]
    })
    return relapses_df, drug_tests_df

def test_define_historical_drug_tests_and_relapses(sample_drug_tests_and_relapses_for_historical_split):
    preprocessor = DataPreprocessorDrugRelapse({})
    relapses_df, drug_tests_df = sample_drug_tests_and_relapses_for_historical_split
    drug_tests_df_result, relapses_df_result = preprocessor._define_historical_drug_tests_and_relapses(relapses_df.copy(), drug_tests_df.copy())
    # For donor 1: last relapse starts at 200, so only the first two tests are historical
    expected = [True, True, False, True, False]
    assert drug_tests_df_result['IS_HISTORICAL_EVENT'].tolist() == expected

    # relapses_df['IS_HISTORICAL_EVENT']: 0 for last event, 1 for others
    expected_relapses = [1, 0, 0]
    assert relapses_df_result['IS_HISTORICAL_EVENT'].tolist() == expected_relapses

def test_define_historical_events():
    relapses_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2, 3, 3, 3],
        'RELAPSE_START': ['2023-01-01', '2023-05-01', '2023-07-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-06-01', '2023-08-01'],
        'IS_LAST_EVENT': [0, 0, 1, 0, 1, 0, 1, 0]
    })

    drug_tests_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2, 3, 3, 3],
        'TIME': ['2023-04-01', '2023-06-15', '2023-08-01', '2023-01-01', '2023-04-01', '2023-02-01', '2023-05-01', '2023-07-01'],
    })

    expected_drug_tests_hist = [True, True, False, True, False, True, True, False]
    expected_relapses_hist = [1, 1, 0, 1, 0, 1, 0, 0]

    preprocessor = DataPreprocessorDrugRelapse({})
    updated_tests, updated_relapses = preprocessor._define_historical_drug_tests_and_relapses(relapses_df, drug_tests_df)

    assert list(updated_tests['IS_HISTORICAL_EVENT']) == expected_drug_tests_hist
    assert list(updated_relapses['IS_HISTORICAL_EVENT']) == expected_relapses_hist

@pytest.fixture
def relapse_positive_merge_data():
    # Drug test history
    drug_tests_df = pd.DataFrame({
        'COLLECTION_ID': [10, 20, 30],
        'POSITIVE_DRUGS': ['cocaine', 'heroin', 'none'],
        'NUM_DRUGS_POSITIVE': [1, 2, 0],
    })
    # Relapses, referencing previous positive test by COLLECTION_ID
    relapses_df = pd.DataFrame({
        'RELAPSE_ID': [1, 2],
        'PREV_POS_COLLECTION_ID': [20, 10],
        'RELAPSE_START': ['2024-07-01', '2024-07-10']
    })
    return relapses_df, drug_tests_df

def test_get_info_positive_before_relapse_start(relapse_positive_merge_data):
    preprocessor = DataPreprocessorDrugRelapse({})
    relapses_df, drug_tests_df = relapse_positive_merge_data
    result = preprocessor._get_info_positive_before_relapse_start(relapses_df, drug_tests_df)

    # For RELAPSE_ID 1, PREV_POS_COLLECTION_ID is 20, info is heroin/2
    # For RELAPSE_ID 2, PREV_POS_COLLECTION_ID is 10, info is cocaine/1
    assert result.loc[result['RELAPSE_ID'] == 1, 'PREV_POSITIVE_DRUGS'].iloc[0] == 'heroin'
    assert result.loc[result['RELAPSE_ID'] == 1, 'PREV_NUM_DRUGS_POSITIVE'].iloc[0] == 2
    assert result.loc[result['RELAPSE_ID'] == 2, 'PREV_POSITIVE_DRUGS'].iloc[0] == 'cocaine'
    assert result.loc[result['RELAPSE_ID'] == 2, 'PREV_NUM_DRUGS_POSITIVE'].iloc[0] == 1

def test_get_num_prev_relapses():
    preprocessor = DataPreprocessorDrugRelapse({})
    relapses_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2],
        'COLLECTION_ID': [101, 102, 103, 201, 202],
        'RELAPSE_START': pd.to_datetime([
            '2020-01-01',  # donor 1 - relapse
            '2020-06-01',  # donor 1 - not relapse
            '2021-01-01',  # donor 1 - relapse
            '2020-03-01',  # donor 2 - relapse
            '2020-09-01'   # donor 2 - relapse
        ]),
        'RELAPSE_EVENT': [1, 0, 1, 1, 1]
    })
    relapses_df = preprocessor._get_num_prev_relapses(relapses_df)

    expected_prev_relapses_s = pd.Series(
         data=[0, 0, 1, 0, 1],
         name='NUM_PREV_RELAPSES',
    )
    assert relapses_df['NUM_PREV_RELAPSES'].equals(expected_prev_relapses_s)

def test_get_past_relapses_time_stats():
    preprocessor = DataPreprocessorDrugRelapse({'binary_event_col': 'RELAPSE_30_DAYS'})
    relapses_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2],
        'COLLECTION_ID': [101, 102, 103, 201, 202],
        'LOG_TIME_UNTIL_NEXT_POSITIVE': [10.0, 20.0, 30.0, 15.0, 25.0],
        'RELAPSE_30_DAYS': [1, 0, 1, 0, 1]
    })
    relapses_df = preprocessor._get_past_relapses_time_stats(relapses_df)
    expected_past_stats_df = pd.DataFrame({
        'DONOR_ID': [1, 1, 1, 2, 2],
        'COLLECTION_ID': [101, 102, 103, 201, 202],
        'LOG_TIME_UNTIL_NEXT_POSITIVE': [10.0, 20.0, 30.0, 15.0, 25.0],
        'RELAPSE_30_DAYS': [1, 0, 1, 0, 1],
        'LOG_TIME_RELAPSE_PAST_MEAN': [np.nan, 10.0, 15.0, np.nan, 15.0],
        'LOG_TIME_RELAPSE_PAST_MEDIAN': [np.nan, 10.0, 15.0, np.nan, 15.0],
        'LOG_TIME_RELAPSE_PAST_STD': [np.nan, np.nan, 7.07106781, np.nan, np.nan],
        'PREV_RELAPSE_30_DAYS': [0, 1, 0, 0, 0],
        'RELAPSE_30_DAYS_PAST_MEAN': [np.nan, 1.0, 0.5, np.nan, 0.0],
        'RELAPSE_30_DAYS_PAST_SUM': [np.nan, 1.0, 1.0, np.nan, 0.0]
    })
    assert_frame_equal(
        relapses_df.reset_index(drop=True),
        expected_past_stats_df.reset_index(drop=True),
        check_exact=False,
        atol=1e-6  # or a larger tolerance if needed
    )