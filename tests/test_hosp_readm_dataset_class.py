import numpy as np
import pandas as pd
import pytest

from recurrent_health_events_prediction.datasets.HospReadmDataset import HospReadmDataset

def _row(subj, order, f1, f2, y, next_type):
    return {
        "SUBJECT_ID": subj,
        "ADMITTIME": order,                # sortable visit order
        "HADM_ID": 10_000 + order,         # not used by the dataset logic here
        "READMISSION_30_DAYS": float(y),   # label
        "NEXT_ADMISSION_TYPE": next_type,  # for elective filtering
        "F1": float(f1),
        "F2": float(f2),
    }

@pytest.fixture
def combined_df():
    """
    One CSV with four subjects:
      - subj 101: length=1, non-elective → expect 1 sample (seq_len=1)
      - subj 102: length=1, ELECTIVE     → expect 0 samples (filtered out)
      - subj 103: length=3, no electives → expect 3 samples (seq lens: 1,2,3)
      - subj 104: length=5, no electives → expect 5 samples, but each sequence
                                            is truncated to max_seq_len=3
    """
    # subj 101: 1 visit, non-elective
    s101 = [
        _row(101, 1, 1.0, 2.0, y=0.0, next_type="URGENT"),
    ]

    # subj 102: 1 visit, elective → filtered out when no_elective=True
    s102 = [
        _row(102, 1, 3.0, 4.0, y=1.0, next_type="ELECTIVE"),
    ]

    # subj 103: 3 visits, all non-elective; v1=(10,11), v2=(20,21), v3=(30,31)
    # labels: y1=0, y2=1, y3=0
    s103 = [
        _row(103, 1, 10.0, 11.0, y=0.0, next_type="EMERGENCY"),
        _row(103, 2, 20.0, 21.0, y=1.0, next_type="URGENT"),
        _row(103, 3, 30.0, 31.0, y=0.0, next_type="EMERGENCY"),
    ]

    # subj 104: 5 visits, all non-elective; v_k=(10k, 10k+1)
    # labels: [0,1,0,1,0] just to vary
    s104 = [
        _row(104, 1, 10.0, 11.0, y=0.0, next_type="URGENT"),
        _row(104, 2, 20.0, 21.0, y=1.0, next_type="EMERGENCY"),
        _row(104, 3, 30.0, 31.0, y=0.0, next_type="URGENT"),
        _row(104, 4, 40.0, 41.0, y=1.0, next_type="EMERGENCY"),
        _row(104, 5, 50.0, 51.0, y=0.0, next_type="URGENT"),
    ]

    df = pd.DataFrame(s101 + s102 + s103 + s104)
    # sanity: all needed columns exist
    needed = {
        "SUBJECT_ID", "ADMITTIME", "HADM_ID",
        "READMISSION_30_DAYS", "NEXT_ADMISSION_TYPE",
        "F1", "F2"
    }
    assert needed.issubset(df.columns)
    return df


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "all_cases.csv" # tmp_path is a pathlib.Path, automatically created by pytest
    df.to_csv(p, index=False)
    return str(p)


def _make_ds(csv_path, max_seq_len=3, no_elective=True):
    return HospReadmDataset(
        csv_path=csv_path,
        max_seq_len=max_seq_len,           # truncated to 3
        feature_cols=["F1", "F2"],
        no_elective=no_elective,
        subject_id_col="SUBJECT_ID",
        order_col="ADMITTIME",
        label_col="READMISSION_30_DAYS",
        next_admt_type_col="NEXT_ADMISSION_TYPE",
        hosp_id_col="HADM_ID",
    )


def test_counts_and_subject_partition(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    # Count samples per subject by peeking at ds.samples (kept by your class)
    from collections import Counter
    counts = Counter(s["subject_id"] for s in ds.samples)

    # subj 101: 1 visit → 1 prefix sample (non-elective)
    assert counts.get(101, 0) == 1
    # subj 102: 1 visit elective → filtered out
    assert counts.get(102, 0) == 0
    # subj 103: 3 visits → 3 prefix samples
    assert counts.get(103, 0) == 3
    # subj 104: 5 visits → 5 prefix samples (each truncated to max_seq_len=3)
    assert counts.get(104, 0) == 5

    # Overall length
    assert len(ds) == 1 + 0 + 3 + 5 == 9


def test_len1_non_elective_case(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    # Get the only sample for subject 101
    s101 = [s for s in ds.samples if s["subject_id"] == 101]
    assert len(s101) == 1
    s = s101[0]
    # one real timestep
    assert s["mask"].sum() == 1
    # most recent first; with len=1 it's the same
    np.testing.assert_allclose(s["x"][0], np.array([1.0, 2.0], dtype=np.float32))
    assert s["y"] == 0.0


def test_len1_elective_filtered(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s102 = [s for s in ds.samples if s["subject_id"] == 102]
    assert len(s102) == 0  # filtered out by no_elective=True


def test_len3_no_electives_prefixes_and_order(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s103 = [s for s in ds.samples if s["subject_id"] == 103]
    assert len(s103) == 3

    # t=1
    seq1 = s103[0]
    assert seq1["mask"].sum() == 1
    np.testing.assert_allclose(seq1["x"][0], np.array([10.0, 11.0], dtype=np.float32))
    assert seq1["y"] == 0.0

    # t=2 (most recent first: v2 at 0, v1 at 1)
    seq2 = s103[1]
    assert seq2["mask"].sum() == 2
    np.testing.assert_allclose(seq2["x"][0], np.array([20.0, 21.0], dtype=np.float32))
    np.testing.assert_allclose(seq2["x"][1], np.array([10.0, 11.0], dtype=np.float32))
    assert seq2["y"] == 1.0

    # t=3 (v3, v2, v1)
    seq3 = s103[2]
    assert seq3["mask"].sum() == 3
    np.testing.assert_allclose(seq3["x"][0], np.array([30.0, 31.0], dtype=np.float32))
    np.testing.assert_allclose(seq3["x"][1], np.array([20.0, 21.0], dtype=np.float32))
    np.testing.assert_allclose(seq3["x"][2], np.array([10.0, 11.0], dtype=np.float32))
    assert seq3["y"] == 0.0


def test_len5_truncates_to_max3_and_recent_first(tmp_path, combined_df):
    """
    With max_seq_len=3 and subject 104 having 5 visits v1..v5 where:
      v1=(10,11), v2=(20,21), v3=(30,31), v4=(40,41), v5=(50,51)
    We expect 5 prefix samples; their sequence lengths are min(t,3) and the
    content is the last min(t,3) visits in most-recent-first order.
    """
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s104 = [s for s in ds.samples if s["subject_id"] == 104]
    assert len(s104) == 5

    # Helper to check expected rows
    def expect_rows(sample, rows):
        # rows is a list of arrays expected at x[0], x[1], ...
        for i, arr in enumerate(rows):
            np.testing.assert_allclose(sample["x"][i], np.array(arr, dtype=np.float32))

    # t=1: [v1]
    assert s104[0]["mask"].sum() == 1
    expect_rows(s104[0], [[10.0, 11.0]])
    # t=2: [v2, v1]
    assert s104[1]["mask"].sum() == 2
    expect_rows(s104[1], [[20.0, 21.0], [10.0, 11.0]])
    # t=3: [v3, v2, v1]
    assert s104[2]["mask"].sum() == 3
    expect_rows(s104[2], [[30.0, 31.0], [20.0, 21.0], [10.0, 11.0]])
    # t=4: truncated to last 3 → [v4, v3, v2]
    assert s104[3]["mask"].sum() == 3
    expect_rows(s104[3], [[40.0, 41.0], [30.0, 31.0], [20.0, 21.0]])
    # t=5: truncated to last 3 → [v5, v4, v3]
    assert s104[4]["mask"].sum() == 3
    expect_rows(s104[4], [[50.0, 51.0], [40.0, 41.0], [30.0, 31.0]])

    # Optional: check labels came through (0,1,0,1,0)
    ys = [s["y"] for s in s104]
    assert ys == [0.0, 1.0, 0.0, 1.0, 0.0]
