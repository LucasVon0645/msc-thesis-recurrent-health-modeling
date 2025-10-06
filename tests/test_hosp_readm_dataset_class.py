import numpy as np
import pandas as pd
import pytest
import torch

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
      - subj 101: length=1, non-elective → expect 1 sample (past seq_len=0, x_current=v1)
      - subj 102: length=1, ELECTIVE     → expect 0 samples (filtered out)
      - subj 103: length=3, no electives → expect 3 samples (past lens: 0,1,2)
      - subj 104: length=5, no electives → expect 5 samples, past truncated to max_seq_len=3
    """
    s101 = [
        _row(101, 1, 1.0, 2.0, y=0.0, next_type="URGENT"),
    ]
    s102 = [
        _row(102, 1, 3.0, 4.0, y=1.0, next_type="ELECTIVE"),
    ]
    s103 = [
        _row(103, 1, 10.0, 11.0, y=0.0, next_type="EMERGENCY"),
        _row(103, 2, 20.0, 21.0, y=1.0, next_type="URGENT"),
        _row(103, 3, 30.0, 31.0, y=0.0, next_type="EMERGENCY"),
    ]
    s104 = [
        _row(104, 1, 10.0, 11.0, y=0.0, next_type="URGENT"),
        _row(104, 2, 20.0, 21.0, y=1.0, next_type="EMERGENCY"),
        _row(104, 3, 30.0, 31.0, y=0.0, next_type="URGENT"),
        _row(104, 4, 40.0, 41.0, y=1.0, next_type="EMERGENCY"),
        _row(104, 5, 50.0, 51.0, y=0.0, next_type="URGENT"),
    ]

    df = pd.DataFrame(s101 + s102 + s103 + s104)
    needed = {
        "SUBJECT_ID", "ADMITTIME", "HADM_ID",
        "READMISSION_30_DAYS", "NEXT_ADMISSION_TYPE",
        "F1", "F2"
    }
    assert needed.issubset(df.columns)
    return df


def _write_csv(tmp_path, df: pd.DataFrame) -> str:
    p = tmp_path / "all_cases.csv"
    df.to_csv(p, index=False)
    return str(p)


def _make_ds(csv_path, max_seq_len=3, no_elective=True):
    return HospReadmDataset(
        csv_path=csv_path,
        max_seq_len=max_seq_len,           # truncated to 3
        longitudinal_feat_cols=["F1", "F2"],  # used for x_past
        current_feat_cols=["F1", "F2"],       # used for x_current
        no_elective=no_elective,
        subject_id_col="SUBJECT_ID",
        order_col="ADMITTIME",
        label_col="READMISSION_30_DAYS",
        next_admt_type_col="NEXT_ADMISSION_TYPE",
        hosp_id_col="HADM_ID",
        # reverse_chronological_order=True is default in the new dataset
    )


def test_counts_and_subject_partition(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    from collections import Counter
    counts = Counter(s["subject_id"] for s in ds.samples)

    assert counts.get(101, 0) == 1   # 1 visit → 1 sample
    assert counts.get(102, 0) == 0   # elective filtered
    assert counts.get(103, 0) == 3   # 3 visits → 3 samples
    assert counts.get(104, 0) == 5   # 5 visits → 5 samples
    assert len(ds) == 9


def test_len1_non_elective_case(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s101 = [s for s in ds.samples if s["subject_id"] == 101]
    assert len(s101) == 1
    s = s101[0]

    # past is empty; current holds v1
    assert s["mask_past"].sum() == 0
    np.testing.assert_allclose(s["x_current"], np.array([1.0, 2.0], dtype=np.float32))
    # x_past is all zeros (padded)
    assert s["x_past"].shape == (3, 2)
    assert s["y"] == 0.0


def test_len1_elective_filtered(tmp_path, combined_df):
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s102 = [s for s in ds.samples if s["subject_id"] == 102]
    assert len(s102) == 0  # filtered out by no_elective=True


def test_len3_no_electives_past_and_current(tmp_path, combined_df):
    """
    Subject 103 visits:
      v1=(10,11), v2=(20,21), v3=(30,31)
    For past-only x_past (most-recent-first among PAST) and separate x_current:
      t=1: x_past=[],               x_current=v1
      t=2: x_past=[v1],             x_current=v2
      t=3: x_past=[v2, v1],         x_current=v3
    """
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s103 = [s for s in ds.samples if s["subject_id"] == 103]
    assert len(s103) == 3

    # t=1
    seq1 = s103[0]
    assert seq1["mask_past"].sum() == 0
    np.testing.assert_allclose(seq1["x_current"], np.array([10.0, 11.0], dtype=np.float32))
    assert seq1["y"] == 0.0

    # t=2
    seq2 = s103[1]
    assert seq2["mask_past"].sum() == 1
    np.testing.assert_allclose(seq2["x_past"][0], np.array([10.0, 11.0], dtype=np.float32))
    np.testing.assert_allclose(seq2["x_current"], np.array([20.0, 21.0], dtype=np.float32))
    assert seq2["y"] == 1.0

    # t=3
    seq3 = s103[2]
    assert seq3["mask_past"].sum() == 2
    np.testing.assert_allclose(seq3["x_past"][0], np.array([20.0, 21.0], dtype=np.float32))
    np.testing.assert_allclose(seq3["x_past"][1], np.array([10.0, 11.0], dtype=np.float32))
    np.testing.assert_allclose(seq3["x_current"], np.array([30.0, 31.0], dtype=np.float32))
    assert seq3["y"] == 0.0


def test_len5_truncates_to_max3_past_only(tmp_path, combined_df):
    """
    Subject 104 visits:
      v1=(10,11), v2=(20,21), v3=(30,31), v4=(40,41), v5=(50,51)
    With max_seq_len=3 and past-only x_past (most-recent-first among PAST):
      t=1: x_past=[],                   x_current=v1
      t=2: x_past=[v1],                 x_current=v2
      t=3: x_past=[v2, v1],             x_current=v3
      t=4: x_past=[v3, v2, v1],         x_current=v4
      t=5: x_past=[v4, v3, v2],         x_current=v5
    """
    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    s104 = [s for s in ds.samples if s["subject_id"] == 104]
    assert len(s104) == 5

    def expect_rows(sample, rows):
        for i, arr in enumerate(rows):
            np.testing.assert_allclose(sample["x_past"][i], np.array(arr, dtype=np.float32))

    # t=1
    assert s104[0]["mask_past"].sum() == 0
    np.testing.assert_allclose(s104[0]["x_current"], np.array([10.0, 11.0], dtype=np.float32))
    # t=2
    assert s104[1]["mask_past"].sum() == 1
    expect_rows(s104[1], [[10.0, 11.0]])
    np.testing.assert_allclose(s104[1]["x_current"], np.array([20.0, 21.0], dtype=np.float32))
    # t=3
    assert s104[2]["mask_past"].sum() == 2
    expect_rows(s104[2], [[20.0, 21.0], [10.0, 11.0]])
    np.testing.assert_allclose(s104[2]["x_current"], np.array([30.0, 31.0], dtype=np.float32))
    # t=4 (truncated to last 3 past visits)
    assert s104[3]["mask_past"].sum() == 3
    expect_rows(s104[3], [[30.0, 31.0], [20.0, 21.0], [10.0, 11.0]])
    np.testing.assert_allclose(s104[3]["x_current"], np.array([40.0, 41.0], dtype=np.float32))
    # t=5
    assert s104[4]["mask_past"].sum() == 3
    expect_rows(s104[4], [[40.0, 41.0], [30.0, 31.0], [20.0, 21.0]])
    np.testing.assert_allclose(s104[4]["x_current"], np.array([50.0, 51.0], dtype=np.float32))

    ys = [s["y"] for s in s104]
    assert ys == [0.0, 1.0, 0.0, 1.0, 0.0]


def test_getitem_tuple_shapes_and_values(tmp_path, combined_df):
    """
    Ensure __getitem__ returns (x_current, x_past, mask_past, y)
    and values match ds.samples for a couple of indices.
    """

    csv_path = _write_csv(tmp_path, combined_df)
    ds = _make_ds(csv_path, max_seq_len=3)

    # pick the second sample of subj 103 (t=2)
    idx = next(i for i, s in enumerate(ds.samples) if s["subject_id"] == 103 and s["t_index"] == 2)
    x_curr, x_past, mask_past, y = ds[idx]

    s = ds.samples[idx]
    assert isinstance(x_curr, torch.Tensor) and x_curr.shape == (2,)
    assert isinstance(x_past, torch.Tensor) and x_past.shape == (3, 2)
    assert isinstance(mask_past, torch.Tensor) and mask_past.shape == (3,)
    assert isinstance(y, torch.Tensor) and y.shape == ()

    np.testing.assert_allclose(x_curr.numpy(), s["x_current"])
    np.testing.assert_allclose(x_past.numpy(), s["x_past"])
    np.testing.assert_array_equal(mask_past.numpy(), s["mask_past"])
    assert float(y.item()) == s["y"]
