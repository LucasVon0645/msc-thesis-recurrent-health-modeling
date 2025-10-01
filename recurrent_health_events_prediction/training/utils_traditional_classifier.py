from typing import Optional, Dict
import neptune
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from recurrent_health_events_prediction.model.utils import plot_model_feature_importance
from recurrent_health_events_prediction.training.utils import (
    plot_permutation_importance,
    plot_model_shap_feature_importance,
)
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_plot_to_neptune_run,
)


def add_training_data_stats_to_neptune(
    neptune_run: neptune.Run,
    training_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    neptune_path="traditional_classifier/training_data",
):
    X = training_df[feature_cols]
    y = training_df[target_col].astype(int)
    neptune_run[f"{neptune_path}/num_samples"] = len(X)
    neptune_run[f"{neptune_path}/num_classes"] = len(np.unique(y))
    neptune_run[f"{neptune_path}/class_distribution"] = (
        pd.Series(y).value_counts(normalize=True).to_dict()
    )
    neptune_run[f"{neptune_path}/feature_names"] = X.columns.tolist()


def scale_features(X_train, X_test, features_not_to_scale: Optional[list[str]] = None):
    scaler = StandardScaler()
    if features_not_to_scale is not None:
        # Identify features to scale
        features_to_scale = [
            col for col in X_train.columns if col not in features_not_to_scale
        ]
    else:
        features_to_scale = X_train.columns.tolist()
    print(f"Scaling features: {features_to_scale}")
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])
    return X_train_scaled, X_test_scaled


def plot_all_feature_importances(
    model,
    X,
    y,
    model_name: str,
    neptune_run: Optional[neptune.Run] = None,
    neptune_path="traditional_classifier/feat_importances",
    random_state=42,
    plot_shap=True,
    show_plots=False,
):
    title = f"{model_name} Feature Importances"
    if not isinstance(model, LogisticRegression):
        fig = plot_model_feature_importance(model, title=title, show_plot=show_plots)
        # Log to Neptune if applicable
        if neptune_run:
            filename = f"{model_name.lower().replace(' ', '_')}_feat_import.png"
            add_plot_to_neptune_run(neptune_run, filename, fig, neptune_path)
    n_repeats = 10
    title = f"{model_name} Permutation Feature Importance - {n_repeats} Repeats"
    _, fig = plot_permutation_importance(
        model,
        X,
        y,
        title=title,
        random_state=random_state,
        n_repeats=n_repeats,
        show_plot=show_plots,
    )

    # Log to Neptune if applicable
    if neptune_run:
        filename = f"{model_name.lower().replace(' ', '_')}_permutation_feat_import.png"
        add_plot_to_neptune_run(neptune_run, filename, fig, neptune_path)

    if plot_shap:
        if isinstance(model, LogisticRegression):
            explainer_type = "linear"
        else:
            explainer_type = "tree"

        X_shap = X.sample(
            min(len(X), 3000), random_state=random_state
        )  # Sample for SHAP to speed up computation
        print(f"Calculating SHAP values for {len(X_shap)} samples...")
        title = f"{model_name} SHAP Feature Importance - {len(X_shap)} training samples"
        fig = plot_model_shap_feature_importance(
            model, X_shap, title=title, explainer=explainer_type, show_plot=show_plots
        )
        if neptune_run:
            filename = f"{model_name.lower().replace(' ', '_')}_shap_feat_import.png"
            add_plot_to_neptune_run(neptune_run, filename, fig, neptune_path)


def add_cv_and_evaluation_results_to_neptune(
    neptune_run: neptune.Run,
    model_name,
    cv_results,
    roc_auc_score,
    neptune_path="traditional_classifier",
):
    neptune_run[f"{neptune_path}/{model_name.lower().replace(' ', '_')}/cv_results"] = (
        cv_results
    )
    neptune_run[
        f"{neptune_path}/{model_name.lower().replace(' ', '_')}/eval_roc_auc_score"
    ] = roc_auc_score


def impute_missing_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, missing_features: dict
):
    """
    Imputes missing values in X_train and X_test in-place using feature-specific strategies.

    Parameters:
    - X_train: Training DataFrame (modified in-place)
    - X_test: Testing DataFrame (modified in-place)
    - missing_features: Dict mapping feature names to imputation strategies
    """
    for col, strategy in missing_features.items():
        if col not in X_train.columns or col not in X_test.columns:
            print(f"Warning: Column '{col}' not found in training or testing data. Skipping...")
            continue

        # Check if the feature contains NaN or NA
        if X_train[col].isna().any() or X_test[col].isna().any():
            print(f"'NA' found in feature '{col}'. Filling with strategy '{strategy}'.")
        else:
            print(f"No 'NA' found in feature '{col}'. No imputation needed.")
            continue

        imputer = SimpleImputer(strategy=strategy)

        # Fit on training data
        imputer.fit(X_train[[col]])

        # Transform and assign back (1D flattening required)
        X_train[col] = imputer.transform(X_train[[col]]).ravel()
        X_test[col] = imputer.transform(X_test[[col]]).ravel()

    return X_train, X_test

def save_prob_predictions(
    out_path: str | Path,
    id_series: pd.Series | np.ndarray,             # test IDs aligned to predictions
    y_true: pd.Series | np.ndarray,   # true labels (test)
    proba_dict: Dict[str, np.ndarray],# e.g. {"logreg": y_pred_proba_logreg, "rf": y_pred_proba_rf, "lgbm": y_pred_proba_lgbm}
    file_format: str = "csv"      # "parquet" or "csv"
) -> pd.DataFrame:
    """
    Create and save a wide per-sample table with ID, y_true, and model probabilities.
    Returns the DataFrame for convenience.
    """
    df = pd.DataFrame({
        "sample_id": np.asarray(id_series, dtype=int),      # column name can be anything; keep consistent
        "y_true": np.asarray(y_true, dtype=int),
    })
    # attach each proba column
    for k, v in proba_dict.items():
        v = np.asarray(v).reshape(-1)
        if len(v) != len(df):
            raise ValueError(f"Length mismatch for '{k}': {len(v)} vs {len(df)}")
        df[f"y_pred_proba_{k}"] = v

    out_path = Path(out_path)

    if file_format.lower() == "csv" or str(out_path).lower().endswith(".csv"):
        df.to_csv(out_path if out_path.suffix else out_path.with_suffix(".csv"), index=False)
    else:
        # default parquet
        if not out_path.suffix:
            out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)

    return df
