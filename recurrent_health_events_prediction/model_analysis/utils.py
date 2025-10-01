import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, log_loss,
    precision_score, recall_score, balanced_accuracy_score
)
from lifelines.utils import concordance_index
from typing import Union, List
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config
from tqdm import tqdm
from enum import Enum

class MetricEnum(Enum):
    ACC = "acc"
    F1 = "f1"
    AUC = "auc"
    LOGLOSS = "logloss"
    PRECISION = "precision"
    RECALL = "recall"
    C_INDEX = "c_index"

## Classification Specific Functions

def add_pred_cols(df, threshold):
    for col in df.columns:
        if 'y_pred_proba_' in col:
            new_col_name = col.replace('y_pred_proba_', 'y_pred_')
            df[new_col_name] = (df[col] >= threshold).astype(int)
    return df

def bin_numeric_column(
    df: pd.DataFrame,
    col: str,
    num_bins: int = 3,
    strategy: str = "quantile",
    labels: bool | list | None = None,
    new_col_name: str | None = None,
) -> pd.DataFrame:
    """
    Bin a numeric column into groups and return a copy of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    col : str
        Name of numeric column to bin.
    num_bins : int
        Number of bins to create.
    strategy : str, {"quantile", "uniform"}
        - "quantile": use pd.qcut (equal-sized groups by data distribution).
        - "uniform": use pd.cut (equal-width intervals).
    labels : bool | list | None
        Labels for bins.
        - False: integer codes 0..num_bins-1.
        - list: custom labels (must match num_bins).
        - None/True: keep interval objects as labels.
    new_col_name : str | None
        Name for new column. Defaults to f"{col}_SUBGROUP".

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with new binned column added.
    """
    df_out = df.copy()
    new_col_name = new_col_name or f"{col}_SUBGROUP"

    if strategy == "quantile":
        df_out[new_col_name] = pd.qcut(
            df_out[col],
            q=num_bins,
            labels=labels,
            duplicates="drop"
        )
    elif strategy == "uniform":
        df_out[new_col_name] = pd.cut(
            df_out[col],
            bins=num_bins,
            labels=labels
        )
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    return df_out

def subset_classifier_metric_report(
    df: pd.DataFrame,
    group_by: Union[str, List[str]],
    metric: str,                       # "acc", "f1", "auc", or "logloss"
    y_col: str,                        # ground truth column (0/1)
    base_col: str,                     # baseline predictions or probabilities
    hmm_col: str,                      # HMM predictions or probabilities
    threshold: float = 0.5,            # used if metric in {"acc","f1"} and inputs are probabilities
    pos_label: int = 1,
    min_pos_neg: int = 5               # minimum positives & negatives per group for AUC/logloss
) -> pd.DataFrame:
    """
    Compute a chosen metric for baseline vs HMM per subgroup, and return a tidy report.

    - For metric in {"acc", "f1"}:
        base_col and hmm_col may be *hard labels* (0/1) or *probabilities*; we threshold probs.
    - For metric in {"auc", "logloss"}:
        base_col and hmm_col should be *probabilities* for the positive class.

    Returns a DataFrame with:
      [group_by..., "n", f"{metric}_baseline", f"{metric}_hmm", "delta"]
    """
    if isinstance(group_by, str):
        group_by = [group_by]
    metric = metric.lower()
    if metric not in {"acc", "f1", "auc", "logloss"}:
        raise ValueError("metric must be one of {'acc','f1','auc','logloss'}")

    def _is_prob_like(s: pd.Series) -> bool:
        # heuristic: float dtype and values mostly within [0,1]
        if not np.issubdtype(s.dtype, np.number):
            return False
        vals = s.dropna().values
        if len(vals) == 0:
            return False
        return np.isfinite(vals).all() and (vals.min() >= 0.0) and (vals.max() <= 1.0)

    def _pred_from_col(y_true: pd.Series, col: pd.Series) -> np.ndarray:
        # turn a column (probs or hard labels) into hard labels
        if _is_prob_like(col):
            return (col.values >= threshold).astype(int)
        # assume already hard labels
        return col.astype(int).values

    def _metric(y_true: pd.Series, col: pd.Series, which: str) -> float:
        y = y_true.astype(int).values
        if which == "acc":
            y_hat = _pred_from_col(y_true, col)
            return balanced_accuracy_score(y, y_hat)
        elif which == "f1":
            y_hat = _pred_from_col(y_true, col)
            return float(f1_score(y, y_hat, pos_label=pos_label, zero_division=0, average='weighted'))
        elif which == "auc":
            # need both classes present and enough samples in each
            y_scores = col.values
            pos = (y == pos_label).sum()
            neg = (y != pos_label).sum()
            if pos < min_pos_neg or neg < min_pos_neg:
                return np.nan
            # If someone passed hard labels by mistake, roc_auc_score will still compute,
            # but it's not ideal; we accept it for robustness.
            return float(roc_auc_score(y, y_scores))
        elif which == "logloss":
            y_scores = col.values
            pos = (y == pos_label).sum()
            neg = (y != pos_label).sum()
            if pos < min_pos_neg or neg < min_pos_neg:
                return np.nan
            # labels ensures stability if a class is missing in the slice (but we guard above)
            return float(log_loss(y, y_scores, labels=[0, 1]))
        else:
            raise RuntimeError("unreachable")

    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n": len(g),
            f"{metric}_baseline": _metric(g[y_col], g[base_col], metric),
            f"{metric}_hmm":      _metric(g[y_col], g[hmm_col],  metric),
        })

    report = (
        df.groupby(group_by, dropna=False, group_keys=False)
          .apply(_agg, include_groups=False)
          .reset_index()
    )
    report["delta"] = report[f"{metric}_hmm"] - report[f"{metric}_baseline"]
    return report

def stratified_bootstrap_delta_classification(
    df,
    y_col,
    base_col,
    hmm_col,
    metric="f1",                  # one of {"f1","acc","auc","logloss","precision","recall"}
    threshold=0.5,                # used if metric needs hard labels and cols are probs
    pos_label=1,
    n_boot=5000,
    random_state=42,
    min_pos_neg=5                 # safeguard for auc/logloss
):
    rng = np.random.default_rng(random_state)
    y = df[y_col].astype(int).to_numpy()
    base = df[base_col].to_numpy()
    hmm  = df[hmm_col].to_numpy()

    def is_prob_like(arr):
        return np.issubdtype(arr.dtype, np.number) and arr.min() >= 0 and arr.max() <= 1

    def to_hard(arr):
        return (arr >= threshold).astype(int) if is_prob_like(arr) else arr.astype(int)

    def compute_metric(y_true, x, which):
        if which in {"f1","acc","precision","recall"}:
            y_hat = to_hard(x)
            if which == "f1":
                return f1_score(y_true, y_hat, pos_label=pos_label, zero_division=0, average='weighted')
            elif which == "acc":
                return balanced_accuracy_score(y_true, y_hat)
            elif which == "precision":
                return precision_score(y_true, y_hat, pos_label=pos_label, zero_division=0, average='weighted')
            elif which == "recall":
                return recall_score(y_true, y_hat, pos_label=pos_label, zero_division=0, average='weighted')
        elif which == "auc":
            pos = (y_true == pos_label).sum()
            neg = (y_true != pos_label).sum()
            if pos < min_pos_neg or neg < min_pos_neg:
                return np.nan
            return roc_auc_score(y_true, x)
        elif which == "logloss":
            pos = (y_true == pos_label).sum()
            neg = (y_true != pos_label).sum()
            if pos < min_pos_neg or neg < min_pos_neg:
                return np.nan
            return log_loss(y_true, x, labels=[0,1])
        else:
            raise ValueError(f"Unsupported metric: {which}")

    # observed delta
    obs_base = compute_metric(y, base, metric)
    obs_hmm  = compute_metric(y, hmm, metric)
    obs_delta = obs_hmm - obs_base

    # stratified resampling
    pos_idx = np.flatnonzero(y == pos_label)
    neg_idx = np.flatnonzero(y != pos_label)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    deltas = []
    for _ in tqdm(range(n_boot)):
        s_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        s_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([s_pos, s_neg])

        y_b   = y[idx]
        base_b = base[idx]
        hmm_b  = hmm[idx]

        m_base = compute_metric(y_b, base_b, metric)
        m_hmm  = compute_metric(y_b, hmm_b,  metric)

        if not (np.isnan(m_base) or np.isnan(m_hmm)):
            deltas.append(m_hmm - m_base)

    deltas = np.array(deltas)
    if len(deltas) == 0:
        return {"obs_delta": obs_delta, "delta_mean": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}

    lo, hi = np.quantile(deltas, [0.025, 0.975])
    p_value = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    return {
        "metric": metric,
        "obs_delta": float(obs_delta),
        "delta_mean": float(deltas.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
        "n_boot": int(len(deltas))
    }

def get_probs_pred_hmm_feat_df(model_name: str, base_model_dir: str):
    model_dir = os.path.join(base_model_dir, model_name)
    probs_pred_hmm_feat_df = pd.read_csv(
        os.path.join(model_dir, "prob_predictions.csv")
    )
    pred_cols = [
        col for col in probs_pred_hmm_feat_df.columns if col.startswith("y_pred_")
    ]
    pred_cols_hmm_renamed = ["hmm_" + col for col in pred_cols]
    probs_pred_hmm_feat_df.rename(
        columns=dict(zip(pred_cols, pred_cols_hmm_renamed)), inplace=True
    )
    return probs_pred_hmm_feat_df

def get_probs_pred_baseline_df(baseline_results_dir: str):
    probs_pred_baseline_df = pd.read_csv(
        os.path.join(baseline_results_dir, "prob_predictions.csv")
    )
    pred_cols = [
        col for col in probs_pred_baseline_df.columns if col.startswith("y_pred_")
    ]
    pred_cols_baseline_renamed = ["baseline_" + col for col in pred_cols]
    probs_pred_baseline_df.rename(
        columns=dict(zip(pred_cols, pred_cols_baseline_renamed)), inplace=True
    )
    return probs_pred_baseline_df

def get_compare_probs_pred_df(
    probs_pred_baseline_df: pd.DataFrame,
    probs_pred_hmm_feat_df: pd.DataFrame,
    threshold: float = 0.5,
):
    compare_pred_df = pd.merge(
        probs_pred_baseline_df,
        probs_pred_hmm_feat_df.drop(columns=["y_true"]),
        on="sample_id",
        how="inner",
    )
    compare_pred_df = add_pred_cols(compare_pred_df, threshold=threshold)
    return compare_pred_df

def select_classifier_to_compare(compare_pred_df, classifier: str):
    if classifier not in ["logreg", "rf", "lgbm"]:
        raise ValueError("classifier must be one of 'logreg', 'rf', 'lgbm'")
    cols = ["sample_id", "y_true"] + [
        col for col in compare_pred_df.columns if classifier in col
    ]
    return compare_pred_df[cols]

def get_pred_prob_col_names(model_to_analyze: str):
    hmm_pred_proba_col = "hmm_y_pred_proba_" + model_to_analyze
    baseline_pred_proba_col = "baseline_y_pred_proba_" + model_to_analyze
    hmm_pred_col = "hmm_y_pred_" + model_to_analyze
    baseline_pred_col = "baseline_y_pred_" + model_to_analyze

    return dict(
        hmm_pred_proba_col=hmm_pred_proba_col,
        baseline_pred_proba_col=baseline_pred_proba_col,
        hmm_pred_col=hmm_pred_col,
        baseline_pred_col=baseline_pred_col,
    )

def get_model_config(model_name: str, base_model_dir: str):
    model_config_path = os.path.join(base_model_dir, model_name, f"{model_name}_config.yaml")
    return import_yaml_config(model_config_path)

def get_final_results_classifier_df(results_bootstrap: dict, model_name: str, classifier: str, hmm_config: dict) -> pd.DataFrame:
    n_states = hmm_config["n_states"]
    final_results_dict = dict(model_name=model_name, n_states=n_states, classifier=classifier, **results_bootstrap)
    final_results_df = pd.DataFrame([final_results_dict])
    return final_results_df

def compare_classifier_with_baseline(
    model_name: str,
    base_model_dir: str,
    baseline_results_dir: str,
    classifier: str,
    threshold: float = 0.5,
    metric: str = "auc",
    n_boot: int = 5000,
    random_state: int = 42,
):
    """
    Compare a specified model with a baseline model using a chosen metric.
    """
    hmm_config = get_model_config(model_name, base_model_dir)

    # Load prediction data
    probs_pred_hmm_feat_df = get_probs_pred_hmm_feat_df(model_name, base_model_dir)
    probs_pred_baseline_df = get_probs_pred_baseline_df(baseline_results_dir)

    print(f"Comparing {model_name} with baseline using {metric} metric...")
    print(f"Number of predictions in baseline: {len(probs_pred_baseline_df)}")
    print(f"Number of predictions in {model_name}: {len(probs_pred_hmm_feat_df)}")
    
    # Merge and prepare data
    compare_pred_df = get_compare_probs_pred_df(
        probs_pred_baseline_df, probs_pred_hmm_feat_df, threshold
    )

    print(f"Number of common predictions: {len(compare_pred_df)}")
    
    # Select relevant columns for the specified classifier
    compare_pred_df = select_classifier_to_compare(compare_pred_df, classifier)

    print("Evaluating HMM Features with classifier: ", classifier)
    
    # Get prediction column names
    cols_dict = get_pred_prob_col_names(classifier)
    
    print("Calculating ", metric, " with ", n_boot, " bootstrap samples...")
    print("")
    # Perform stratified bootstrap to compare models
    bootstrap_results = stratified_bootstrap_delta_classification(
        compare_pred_df,
        y_col="y_true",
        base_col=cols_dict["baseline_pred_proba_col"],
        hmm_col=cols_dict["hmm_pred_proba_col"],
        metric=metric,
        threshold=threshold,
        n_boot=n_boot,
        random_state=random_state
    )

    final_results = get_final_results_classifier_df(bootstrap_results, model_name, classifier, hmm_config)
    
    return final_results


## Survival Analysis Specific Functions

def get_partial_hazard_pred_df(results_dir: str, model_name: str, baseline: bool = False):
    model_results_path = os.path.join(results_dir, model_name)
    df = pd.read_csv(
        os.path.join(model_results_path, "partial_hazards.csv")
    )
    if baseline:
        df = df.rename(columns={"partial_hazard": "baseline_partial_hazard"})
    else:
        df = df.rename(columns={"partial_hazard": "hmm_partial_hazard"})
    return df

def get_compare_partial_hazard_df(
    events_df: pd.DataFrame,
    partial_hazard_baseline_df: pd.DataFrame,
    partial_hazard_hmm_feat_df: pd.DataFrame,
    id_col: str = "HADM_ID",
    event_col: str = "READMISSION_EVENT",
    duration_col: str = "EVENT_DURATION"
):
    probs_pred_hmm_feat_df = pd.merge(
        partial_hazard_baseline_df,
        partial_hazard_hmm_feat_df,
        on=id_col,
        how="inner",
    )
    
    probs_pred_hmm_feat_df = pd.merge(
        probs_pred_hmm_feat_df,
        events_df[[id_col, event_col, duration_col]],
        on=id_col,
        how="inner"
    )

    return probs_pred_hmm_feat_df

def stratified_bootstrap_delta_cindex_lifelines(
    df,
    duration_col,            # e.g. "time"
    event_col,               # e.g. "event" (1/0 or True/False)
    base_pred_col,           # predictions for model A (already in "higher=better survival" scale)
    hmm_pred_col,            # predictions for model B (same scale)
    n_boot=5000,
    random_state=42,
    stratify=True,           # resample events and censors separately
    return_boot=False        # whether to return the raw bootstrap deltas
):
    """
    Bootstrap Δ c-index between two fixed sets of predictions using lifelines only.

    IMPORTANT on prediction scale:
    lifelines.concordance_index expects larger values = longer survival.
    - If you have risk scores (larger = higher hazard), pass their negative (as you already do).
    - If you have predicted median survival times, they are already in the right direction.

    Returns a dict with observed c-indices, observed delta, bootstrap CI, and p-value.
    """

    rng = np.random.default_rng(random_state)

    # test-set arrays
    t = df[duration_col].to_numpy()
    e = df[event_col].astype(bool).to_numpy()
    a = -df[base_pred_col].to_numpy()
    b = -df[hmm_pred_col].to_numpy()

    # helper
    def cindex_lfl(t_, e_, s_):
        # lifelines returns a float; may be NaN if there are no comparable pairs
        return float(concordance_index(t_, s_, e_))

    # observed values (no resampling)
    c_base = cindex_lfl(t, e, a)
    c_hmm  = cindex_lfl(t, e, b)
    obs_delta = c_hmm - c_base

    # prepare bootstrap indices
    n = len(t)
    if stratify:
        idx_e = np.flatnonzero(e)
        idx_c = np.flatnonzero(~e)
        ne, nc = len(idx_e), len(idx_c)
        # if one stratum is empty, fall back to non-stratified
        stratify = (ne > 0 and nc > 0)
    else:
        idx_all = np.arange(n)

    deltas = np.empty(n_boot, dtype=float)
    kept = 0

    for _ in tqdm(range(n_boot)):
        if stratify:
            s_e = rng.choice(idx_e, size=ne, replace=True)
            s_c = rng.choice(idx_c, size=nc, replace=True)
            idx = np.concatenate([s_e, s_c])
        else:
            idx = rng.choice(n, size=n, replace=True)

        t_b = t[idx]; e_b = e[idx]
        a_b = a[idx]; b_b = b[idx]

        cA = cindex_lfl(t_b, e_b, a_b)
        cB = cindex_lfl(t_b, e_b, b_b)

        # skip pathologies (no comparable pairs can yield NaN)
        if not (np.isnan(cA) or np.isnan(cB)):
            deltas[kept] = cB - cA
            kept += 1

    if kept == 0:
        out = {
            "metric": MetricEnum.C_INDEX.value,
            "obs_c_base": float(c_base),
            "obs_c_hmm": float(c_hmm),
            "obs_delta": float(obs_delta),
            "delta_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n_boot_kept": 0,
            "n_boot": int(n_boot),
            "stratified": bool(stratify),
        }
        if return_boot:
            out["deltas"] = np.array([])
        return out

    deltas = deltas[:kept]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    # two-sided p from bootstrap sign proportion
    p_value = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())

    out = {
        "metric": MetricEnum.C_INDEX.value,
        "obs_c_base": float(c_base),
        "obs_c_hmm": float(c_hmm),
        "obs_delta": float(obs_delta),
        "delta_mean": float(deltas.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
        "n_boot_kept": int(kept),
        "n_boot": int(n_boot),
        "stratified": bool(stratify),
    }
    if return_boot:
        out["deltas"] = deltas
    return out
