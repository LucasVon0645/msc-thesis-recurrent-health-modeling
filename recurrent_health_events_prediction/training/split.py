import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Iterable

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.model_selection import StratifiedGroupKFold


@dataclass
class SplitSummary:
    table: pd.DataFrame  # per-patient table with metrics & strata
    train_ids: Optional[set] = None
    test_ids: Optional[set] = None
    fold_assignments: Optional[pd.DataFrame] = None  # columns: SUBJECT_ID, fold
    balance_report: Optional[pd.DataFrame] = None    # quick stats per split/fold

num_visits_bins = [0, 1, 2, 3, 5, 10, np.inf]
num_visits_labels = ["1", "2", "3", "4-5", "6-10", "11+"]
mean_readm_bins = [-1e-9, 0, 0.1, 0.3, 0.6, 0.9, 1.0 + 1e-9]
mean_readm_labels = ["0", "(0,0.1]", "(0.1,0.3]", "(0.3,0.6]", "(0.6,0.9]", "(0.9,1]"]
mean_charlson_bins = [-1e-9, 2, 4, 6, 8, 10, np.inf]
mean_charlson_labels = ["0-2", "2-4", "4-6", "6-8", "8-10", "10+"]


def _bucketize(series: pd.Series, bins, labels=None, include_lowest=True) -> pd.Series:
    if labels is None:
        labels = [f"({bins[i]},{bins[i+1]}]" for i in range(len(bins)-1)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=include_lowest, right=True)


def build_patient_table(
    df: pd.DataFrame,
    subject_id_col: str = "SUBJECT_ID",
    label_col: str = "READMISSION_30_DAYS",
    charlson_col: str = "CHARLSON_INDEX",
) -> pd.DataFrame:
    """Aggregate to one row per patient and create bucketed strata."""
    g = df.groupby(subject_id_col)

    out = pd.DataFrame({
        subject_id_col: g.size().index,
        "num_visits": g.size().values,
        "any_readmission": g[label_col].max().astype(int).values,
        "mean_readmission": g[label_col].mean().fillna(0).values,
    })

    if charlson_col in df.columns:
        out["mean_CHARLSON_INDEX"] = g[charlson_col].mean().fillna(0).values
    else:
        out["mean_CHARLSON_INDEX"] = 0.0  # fallback

    # Buckets
    out["b_num_visits"] = _bucketize(out["num_visits"], num_visits_bins, labels=num_visits_labels).astype(str)
    out["b_mean_readmission"] = _bucketize(
        out["mean_readmission"], mean_readm_bins,
        labels=mean_readm_labels
    ).astype(str)
    out["b_mean_charlson"] = _bucketize(out["mean_CHARLSON_INDEX"], mean_charlson_bins, labels=mean_charlson_labels).astype(str)
    out["b_any_readm"] = out["any_readmission"].astype(str)

    # Composite stratification key: keep it compact to avoid sparsity
    out["strata"] = (
        out["b_num_visits"] + "|" +
        out["b_mean_readmission"] + "|" +
        out["b_any_readm"] + "|" +
        out["b_mean_charlson"]
    )

    return out


def _collapse_rare_strata(tbl: pd.DataFrame, min_size: int = 2) -> pd.DataFrame:
    """Collapse strata with < min_size patients into 'OTHER' to avoid stratify errors."""
    counts = tbl["strata"].value_counts()
    rare = counts[counts < min_size].index
    if len(rare) > 0:
        tbl = tbl.copy()
        tbl.loc[tbl["strata"].isin(rare), "strata"] = "OTHER"
    return tbl


def _balance_report(tbl: pd.DataFrame, id_col: str, splits: Dict[str, Iterable]) -> pd.DataFrame:
    """Quick per-split summary to sanity-check similarity."""
    rows = []
    for name, ids in splits.items():
        sub = tbl[tbl[id_col].isin(set(ids))].copy()
        rows.append(pd.Series({
            "patients": len(sub),
            "num_visits_mean": sub["num_visits"].mean(),
            "num_visits_median": sub["num_visits"].median(),
            "any_readmission_rate": sub["any_readmission"].mean(),
            "mean_readmission_mean": sub["mean_readmission"].mean(),
            "mean_charlson_mean": sub["mean_CHARLSON_INDEX"].mean(),
        }, name=name))
    return pd.DataFrame(rows)


def split_train_test_stratified_group(
    df: pd.DataFrame,
    subject_id_col: str = "SUBJECT_ID",
    label_col: str = "READMISSION_30_DAYS",
    charlson_col: str = "CHARLSON_INDEX",
    test_size: float = 0.2,
    random_state: int = 42,
    min_stratum_size: int = 2,
) -> SplitSummary:
    """
    Patient-disjoint train/test split with stratification on a composite of:
    num_visits, mean_readmission, any_readmission, mean_CHARLSON_INDEX (bucketed).
    """
    tbl = build_patient_table(
        df,
        subject_id_col,
        label_col,
        charlson_col
    )
    tbl = _collapse_rare_strata(tbl, min_size=min_stratum_size)

    ids = tbl[subject_id_col]
    y = tbl["strata"]
    # Since tbl is one row per patient, splitting here is patient-disjoint by construction.
    strat = y if y.nunique() > 1 else None

    train_ids, test_ids = train_test_split(
        ids, test_size=test_size, random_state=random_state, stratify=strat
    )
    train_ids, test_ids = set(train_ids), set(test_ids)

    report = _balance_report(tbl, subject_id_col, {"train": train_ids, "test": test_ids})
    return SplitSummary(table=tbl, train_ids=train_ids, test_ids=test_ids, balance_report=report)


def make_cv_folds_stratified_group(
    df: pd.DataFrame,
    subject_id_col: str = "SUBJECT_ID",
    label_col: str = "READMISSION_30_DAYS",
    charlson_col: str = "CHARLSON_INDEX",
    n_splits: int = 5,
    random_state: int = 42,
    min_stratum_size: int = 2,
) -> SplitSummary:
    """
    Build fixed, patient-disjoint CV folds attempting to balance:
    num_visits, mean_readmission, any_readmission, mean_CHARLSON_INDEX (bucketed).
    Uses StratifiedGroupKFold if available; falls back to (1) StratifiedKFold on
    the patient table, or (2) GroupKFold without stratification.
    """
    tbl = build_patient_table(
        df,
        subject_id_col,
        label_col,
        charlson_col
    )
    tbl = _collapse_rare_strata(tbl, min_size=min_stratum_size)

    # One row per patient already
    X = np.zeros((len(tbl), 1))
    y = tbl["strata"]
    groups = tbl[subject_id_col].values

    fold_col = np.full(len(tbl), -1, dtype=int)

    if y.nunique() > 1: # At least 2 strata present, try stratified splits
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X=X, y=y, groups=groups)
    else:
        # Last resort: group-only splits, no stratification
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X=X, y=None, groups=groups)

    for k, (_, val_idx) in enumerate(splits):
        fold_col[val_idx] = k

    folds_df = tbl[[subject_id_col]].copy()
    folds_df["fold"] = fold_col

    # Small balance report per fold
    split_map = {f"fold_{k}": set(folds_df.loc[folds_df["fold"] == k, subject_id_col]) for k in range(n_splits)}
    report = _balance_report(tbl, subject_id_col, split_map)

    return SplitSummary(table=tbl, fold_assignments=folds_df, balance_report=report)
