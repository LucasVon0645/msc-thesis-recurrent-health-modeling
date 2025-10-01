import pandas as pd
import pytest
from unittest.mock import patch
from unittest.mock import MagicMock

from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import (
    RecurrentHealthEventsHMM,
)

import recurrent_health_events_prediction.preprocessing.gen_dataset_hmm as gen_dataset_hmm
from recurrent_health_events_prediction.preprocessing.utils import get_past_events
from recurrent_health_events_prediction.training.utils_hmm import (
    summarize_hidden_state_counts_from_df,
    add_pred_state_to_df,
)


@pytest.fixture
def hmm_model():
    past_sequences = {
        1: [1, 1, 0, 0],  # Subject 1, 4 events: 2 "high" (1), 2 "low" (0)
        2: [],  # Subject 2, no past events
        3: [1, 0],  # Subject 3, 2 events: 1 "high" (1), 1 "low" (0)
        4: [],  # Subject 4, no past events
    }

    def mock_infer_hidden_states(df):
        # Assume df is always sorted/grouped per subject
        return [past_sequences[sid] for sid in df["SUBJECT_ID"].unique()]

    mock_model = MagicMock()
    mock_model.infer_hidden_states.side_effect = mock_infer_hidden_states
    mock_model.get_hidden_state_labels.return_value = {0: "low", 1: "high"}
    return mock_model


@pytest.fixture
def test_dfs():
    # last_events_with_hmm_features_df: just the last event per subject
    last_events_with_hmm_features_df = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4],  # ids must be consistent with past events
            "EVENT_ID": [5, 6, 9, 10],  # just for debugging, not used bby the function
            "OTHER_COL": [10, 20, 30, 40],
            "EVENT": [1, 1, 0, 0],  # can be anything
        }
    )

    # Create past events for each subject. These dataframes must contain the last event as well.
    events_up_to_last_obs_df = pd.DataFrame(
        {
            "SUBJECT_ID": [1] * 5
            + [2] * 1,  # id must be consistent with last_events_with_hmm_features_df
            "EVENT_ID": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],  # just for debugging, not used bby the function
            "EVENT": [1] * 6,  # could be anything, not used in test
        }
    )
    events_up_to_last_censored_df = pd.DataFrame(
        {
            "SUBJECT_ID": [3] * 3
            + [4] * 1,  # id must be consistent with last_events_with_hmm_features_df
            "EVENT_ID": [7, 8, 9, 10],  # just for debugging, not used bby the function
            "EVENT": [1] * 2 + [0] * 1 + [0] * 1,  # could be anything, not used in test
        }
    )

    return (
        last_events_with_hmm_features_df,
        events_up_to_last_obs_df,
        events_up_to_last_censored_df,
    )


def test_infer_past_states_add_stats(hmm_model, test_dfs):
    (
        last_events_with_hmm_features_df,
        events_up_to_last_obs_df,
        events_up_to_last_censored_df,
    ) = test_dfs

    result = gen_dataset_hmm.infer_past_states_add_stats(
        hmm_model,
        last_events_with_hmm_features_df,
        events_up_to_last_obs_df,
        events_up_to_last_censored_df,
        subject_id_col="SUBJECT_ID",
    )

    # Now check the result. You should expect "PAST_COUNT_HIDDEN_RISK_LOW" and "PAST_COUNT_HIDDEN_RISK_HIGH" columns
    assert "PAST_COUNT_HIDDEN_RISK_LOW" in result.columns
    assert "PAST_COUNT_HIDDEN_RISK_HIGH" in result.columns
    assert (
        "HEALTH_HIDDEN_STATE" not in result.columns
    )  # This column should not be present

    expected_df = pd.DataFrame(
        {
            "SUBJECT_ID": [1, 2, 3, 4],
            "EVENT_ID": [5, 6, 9, 10],
            "OTHER_COL": [10, 20, 30, 40],
            "EVENT": [1, 1, 0, 0],
            "PAST_COUNT_HIDDEN_RISK_HIGH": [2, 0, 1, 0],
            "PAST_COUNT_HIDDEN_RISK_LOW": [2, 0, 1, 0],
        }
    )

    pd.testing.assert_frame_equal(result, expected_df)


def test_extract_partial_trajectories():
    all_events_df = pd.DataFrame({
        "SUBJECT_ID": [1, 1, 2, 2, 3, 3, 4],
        "HADM_ID":    [10, 11, 20, 21, 30, 31, 40],
        "ADMITTIME": pd.to_datetime([
            "2020-01-01", "2020-01-03",
            "2020-01-02", "2020-01-06",
            "2020-01-03", "2020-01-08",
            "2020-01-09"  # Subject 4, only one row
        ]),
        "READMISSION_30_DAYS": [5.0, 2.0, 7.0, 1.0, 6.0, 3.0, 9.0]
    }).sample(frac=1).reset_index(drop=True)  # unordered input

    last_events_df = pd.DataFrame({
        "SUBJECT_ID": [1, 2, 3, 4],
        "HADM_ID": [11, 21, 31, 40],
        "READMISSION_EVENT": [1, 0, 1, 1],
        "READMISSION_30_DAYS": [2.0, 1.0, 3.0, 9.0]  # also exists here
    }).sample(frac=1).reset_index(drop=True)

    obs_ids, cens_ids, obs_df, cens_df = gen_dataset_hmm.extract_partial_trajectories(
        all_events_df,
        last_events_df,
        time_feature_col="READMISSION_30_DAYS",
        event_name="readmission",
        event_id_col="HADM_ID",
        subject_id_col="SUBJECT_ID",
        time_col="ADMITTIME"
    )

    # Subject 4 should be in observed
    assert 4 in obs_ids.index
    assert 40 in obs_df[obs_df["SUBJECT_ID"] == 4]["HADM_ID"].values

    # The one row for Subject 4 must be masked
    subject_4_row = obs_df[(obs_df["SUBJECT_ID"] == 4) & (obs_df["HADM_ID"] == 40)]
    assert len(subject_4_row) == 1
    assert pd.isna(subject_4_row.iloc[0]["READMISSION_30_DAYS"]), "Subject 4's only row should be masked"

    # Sanity check for other subjects
    for subject_id, hadm_id in obs_ids.items():
        match = (obs_df["SUBJECT_ID"] == subject_id) & (obs_df["HADM_ID"] == hadm_id)
        assert obs_df.loc[match, "READMISSION_30_DAYS"].isna().all(), f"READMISSION_30_DAYS not masked for observed SUBJECT_ID={subject_id}"
    
    for subject_id, hadm_id in cens_ids.items():
        match = (cens_df["SUBJECT_ID"] == subject_id) & (cens_df["HADM_ID"] == hadm_id)
        assert cens_df.loc[match, "READMISSION_30_DAYS"].isna().all(), f"READMISSION_30_DAYS not masked for censored SUBJECT_ID={subject_id}"

    # Ensure earlier rows are unmasked
    for sid in obs_df["SUBJECT_ID"].unique():
        subj = obs_df[obs_df["SUBJECT_ID"] == sid].sort_values("ADMITTIME")
        if len(subj) > 1:
            assert subj.iloc[:-1]["READMISSION_30_DAYS"].notna().all()
