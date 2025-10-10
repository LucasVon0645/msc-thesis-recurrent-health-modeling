import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
from typing import Optional, Sequence, Tuple
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from plotly import graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

def summarize_search_results(search_cv, print_results: bool = True, model_name: str = "Model"):
    """
    Summarize the best results from a fitted RandomizedSearchCV or GridSearchCV object.
    
    Parameters:
        search_cv: fitted RandomizedSearchCV or GridSearchCV object
        
    Returns:
        dict with best_params, best_score_mean, best_score_std
    """
    best_index = search_cv.best_index_
    best_score_mean = search_cv.cv_results_['mean_test_score'][best_index].item()
    best_score_std = search_cv.cv_results_['std_test_score'][best_index].item()
    num_folds = search_cv.n_splits_
    mean_scores = search_cv.cv_results_['mean_test_score']
    n_fitted_candidates = len(mean_scores) - np.isnan(mean_scores).sum()

    if print_results:
        print(f"{n_fitted_candidates} candidates of {model_name.capitalize()} trained and validated with cross-validation on {num_folds} folds.")
        print(f"Best parameters: {search_cv.best_params_}")
        print(f"Validation Score - {model_name.capitalize()}: {best_score_mean:.3f} ± {best_score_std:.3f}")
    
    return {
        "model_name": model_name,
        "best_score_mean": best_score_mean,
        "best_score_std": best_score_std,
        "n_fitted_candidates": n_fitted_candidates,
        "num_folds": num_folds,
        "best_params": search_cv.best_params_
    }

def plot_permutation_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test,
    title: str = 'Permutation Feature Importance',
    n_repeats: int = 10,
    scoring: str = 'roc_auc_ovr',
    random_state: int = 42,
    show_plot: bool = True
):
    """
    Plots the permutation importance of features for a given sklearn estimator,
    using seaborn for the bars and matplotlib for the error bars.
    """
    result = permutation_importance(
        estimator, X_test, y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring
    )

    sorted_idx = result.importances_mean.argsort()[::-1]
    features = X_test.columns[sorted_idx]
    importances = result.importances_mean[sorted_idx]
    stds = result.importances_std[sorted_idx]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Standard Deviation': stds
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        orient='h',
        color='skyblue'
    )
    
    # Add error bars manually
    for i, (importance, std) in enumerate(zip(importance_df['Importance'], importance_df['Standard Deviation'])):
        ax.errorbar(
            x=importance, y=i,
            xerr=std,
            fmt='none',
            c='black',
            capsize=3,
            linewidth=1.5
        )

    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    fig = plt.gcf()  # Get the current figure
    if show_plot:
        plt.show()
    plt.close()  # Close the plot to avoid display in Jupyter notebooks
    return importance_df, fig

def plot_model_shap_feature_importance(model, X, title = "SHAP Feature Importance", explainer='tree', show_plot: bool = False):
    if explainer == 'tree':
        explainer = shap.TreeExplainer(model)
    elif explainer == 'linear':
        explainer = shap.LinearExplainer(model, X)
    elif explainer == 'kernel':
        explainer = shap.KernelExplainer(model.predict, X)
    else:
        raise ValueError("Unsupported explainer type. Use 'tree', 'linear', or 'kernel'.")
    
    shap_values = explainer.shap_values(X).astype(np.float64)

    if shap_values.ndim == 3:  # (n_samples, n_features, n_classes)
        shap_values_to_plot = shap_values[:, :, 1]  # class 1
    else:
        shap_values_to_plot = shap_values  # already (n_samples, n_features)
        
    # Now plot
    plt.title(title, pad=20)
    shap.summary_plot(shap_values_to_plot, X, feature_names=X.columns, show=False)
    fig = plt.gcf()  # Get the current figure
    if show_plot:
        plt.show()
    plt.close()  # Close the plot to avoid display in Jupyter notebooks
    return fig

def one_hot_encode_feature(df: pd.DataFrame, features: list[str], suffix: Optional[str] = None, drop_first: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    One-hot encodes specified features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the features to encode.
        features (list[str]): List of feature names to one-hot encode.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded features.
        list: List of new columns created by one-hot encoding.
    """
    prev_cols = df.columns.tolist()
    df = pd.get_dummies(df, columns=features, drop_first=drop_first)
    if suffix is None:
        suffix = ""
    df.columns = [(col + suffix).upper().replace(" ", "_") for col in df.columns]
    final_cols = df.columns.tolist()
    new_cols = set(final_cols) - set(prev_cols)

    return df, list(new_cols)

def compare_dataframes(df1, df2, key_columns=None):
    """
    Compare two dataframes and return differences.

    Args:
        df1 (pd.DataFrame): First dataframe.
        df2 (pd.DataFrame): Second dataframe.
        key_columns (list, optional): Columns to use for merging/comparing rows.

    Returns:
        dict: Dictionary with differences.
    """
    differences = {}

    # Check columns
    if set(df1.columns) != set(df2.columns):
        differences["column_diff"] = {
            "only_in_df1": list(set(df1.columns) - set(df2.columns)),
            "only_in_df2": list(set(df2.columns) - set(df1.columns))
        }

    # Check shape
    if df1.shape != df2.shape:
        differences["shape_diff"] = {
            "df1_shape": df1.shape,
            "df2_shape": df2.shape
        }

    # Align columns
    common_cols = sorted(set(df1.columns).intersection(df2.columns))
    df1 = df1[common_cols].copy()
    df2 = df2[common_cols].copy()

    # Sort if key columns are provided
    if key_columns:
        df1 = df1.sort_values(by=key_columns).reset_index(drop=True)
        df2 = df2.sort_values(by=key_columns).reset_index(drop=True)

    # Check row-by-row differences
    comparison = df1.compare(df2, keep_shape=True, keep_equal=False)
    if not comparison.empty:
        differences["row_diff"] = comparison

    return differences

def preprocess_features_to_one_hot_encode(
    df: pd.DataFrame,
    features_to_encode: list,
    one_hot_cols_to_drop: Optional[list] = None,
):
    """
    Preprocess features to encode and drop specified values.

    Args:
        df (pd.DataFrame): DataFrame containing features.
        features_to_encode (list): List of features to one-hot encode.
        values_to_drop (list, optional): Values to drop from the DataFrame.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with one-hot encoded features.
    """
    if one_hot_cols_to_drop is not None:
        drop_first = False
    else:
        drop_first = True

    df, new_cols = one_hot_encode_feature(df, features_to_encode, drop_first=drop_first)

    if one_hot_cols_to_drop is not None:
        for col in new_cols:
            if col in one_hot_cols_to_drop:
                df = df.drop(columns=col, errors="ignore")

    # Remove dropped columns from new_cols
    new_cols = sorted(
        [col for col in new_cols if col not in one_hot_cols_to_drop]
        if one_hot_cols_to_drop is not None
        else new_cols
    )

    return df, new_cols

def make_train_test_split_file(
    df: pd.DataFrame,
    id_col: str,
    target_col: Optional[str] = None,  # if None -> no stratification
    test_size: float = 0.2,
    random_state: int = 42,
    out_path: str | Path = "train_test_split.csv",
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Create a single CSV mapping each sample id to 'train' or 'test'.

    Parameters
    ----------
    df : DataFrame
        Full dataset (not pre-split).
    id_col : str
        Name of a unique ID column in df.
    target_col : Optional[str]
        Column to stratify on (classification target). If None, no stratification.
    test_size : float
        Proportion for test split.
    random_state : int
        RNG seed.
    out_path : str | Path
        Where to write the CSV (columns: id_col, split).
    overwrite : bool
        If False and file exists, raise an error.

    Returns
    -------
    split_map : DataFrame
        Two columns: [id_col, 'split'] with values 'train' or 'test'.
    """
    df = df.copy()

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df.")

    # Ensure unique IDs
    if df[id_col].duplicated().any():
        dupes = df[id_col][df[id_col].duplicated()].unique()[:5]
        raise ValueError(f"ID column '{id_col}' contains duplicates, e.g. {dupes!r}.")

    n = len(df)
    indices = np.arange(n)

    if target_col is not None:
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not found in df.")
        y = df[target_col].values
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        (train_idx, test_idx), = splitter.split(np.zeros(n), y)
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        (train_idx, test_idx), = splitter.split(indices)

    # Build split map
    split_map = pd.concat(
        [
            pd.DataFrame({id_col: df.iloc[train_idx][id_col].values, "split": "train"}),
            pd.DataFrame({id_col: df.iloc[test_idx][id_col].values,  "split": "test"}),
        ],
        ignore_index=True,
    )

    # Optional sanity stats
    split_map["split"] = split_map["split"].astype("category")

    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists and overwrite=False.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split_map.to_csv(out_path, index=False)

    return split_map

def apply_train_test_split_file_classification(
    df: pd.DataFrame,
    split_csv_path: str | Path,
    id_col: str,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load a split CSV and apply it to df to get X_train, X_test, y_train, y_test.
    It returns the ids for each split as well.

    Parameters
    ----------
    df : DataFrame
        Full dataset containing id_col, target_col, and features.
    split_csv_path : str | Path
        Path to CSV produced by make_train_test_split_file (columns: id_col, 'split').
    id_col : str
        Unique ID column to merge on.
    target_col : str
        Target column name.
    feature_cols : Optional[Sequence[str]]
        If None, uses all columns except [id_col, target_col].
        Otherwise, uses these exactly (must exist in df).

    Returns
    -------
    X_train, X_test : DataFrame
    y_train, y_test : Series
    train_ids, test_ids : np.ndarray
    """
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in df.")

    split_df = pd.read_csv(split_csv_path)
    expected_cols = {id_col, "split"}
    if not expected_cols.issubset(split_df.columns):
        raise ValueError(
            f"Split file must contain columns {expected_cols}, got {split_df.columns.tolist()}"
        )

    # Validate IDs
    if split_df[id_col].duplicated().any():
        dupes = split_df[id_col][split_df[id_col].duplicated()].unique()[:5]
        raise ValueError(f"Split file has duplicate IDs, e.g. {dupes!r}.")

    # Keep only rows present in df; warn if some are missing
    missing = set(split_df[id_col]) - set(df[id_col])
    if missing:
        # You could raise instead if this should be strict:
        raise ValueError(f"{len(missing)} IDs in split file not found in df, e.g. {list(missing)[:5]}")

    # Merge split labels onto df
    df_merged = df.merge(split_df[[id_col, "split"]], on=id_col, how="inner")

    # Compute features set
    if feature_cols is None:
        feature_cols = [c for c in df_merged.columns if c not in (id_col, target_col, "split")]
    else:
        missing_feats = set(feature_cols) - set(df_merged.columns)
        if missing_feats:
            raise ValueError(f"Missing feature columns: {sorted(missing_feats)}")

    # Build splits
    train_mask = df_merged["split"].eq("train")
    test_mask  = df_merged["split"].eq("test")

    X_train = df_merged.loc[train_mask, feature_cols].copy()
    X_test  = df_merged.loc[test_mask,  feature_cols].copy()
    y_train = df_merged.loc[train_mask, target_col].astype(int).copy()
    y_test  = df_merged.loc[test_mask,  target_col].astype(int).copy()
    train_ids = df_merged.loc[train_mask, id_col].values
    test_ids  = df_merged.loc[test_mask,  id_col].values

    # Optional: preserve original row order within each split by id_col sort
    # X_train = X_train.set_index(df_merged.loc[train_mask, id_col]).sort_index()
    # X_test  = X_test.set_index(df_merged.loc[test_mask, id_col]).sort_index()
    # y_train = y_train.loc[X_train.index]
    # y_test  = y_test.loc[X_test.index]

    return X_train, X_test, y_train, y_test, train_ids, test_ids

def apply_train_test_split_file_survival(
    df: pd.DataFrame,
    split_csv_path: str | Path,
    id_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not found in df.")

    split_df = pd.read_csv(split_csv_path)
    expected_cols = {id_col, "split"}
    if not expected_cols.issubset(split_df.columns):
        raise ValueError(
            f"Split file must contain columns {expected_cols}, got {split_df.columns.tolist()}"
        )
    
    # Validate IDs
    if split_df[id_col].duplicated().any():
        dupes = split_df[id_col][split_df[id_col].duplicated()].unique()[:5]
        raise ValueError(f"Split file has duplicate IDs, e.g. {dupes!r}.")

    # Keep only rows present in df; warn if some are missing
    missing = set(split_df[id_col]) - set(df[id_col])
    if missing:
        # You could raise instead if this should be strict:
        raise ValueError(f"{len(missing)} IDs in split file not found in df, e.g. {list(missing)[:5]}")

    # Merge split labels onto df
    df_merged = df.merge(split_df[[id_col, "split"]], on=id_col, how="inner")

    # Build splits
    train_mask = df_merged["split"].eq("train")
    test_mask  = df_merged["split"].eq("test")

    X_train = df_merged.loc[train_mask].drop(columns=["split"])
    X_test  = df_merged.loc[test_mask].drop(columns=["split"])
    train_ids = df_merged.loc[train_mask, id_col].values
    test_ids  = df_merged.loc[test_mask,  id_col].values

    return X_train, X_test, train_ids, test_ids

def standard_scale_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    save_scaler_dir_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if feature_cols is None:
        feature_cols = X_train.columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    if save_scaler_dir_path is not None:
        filepath = os.path.join(save_scaler_dir_path, 'scaler.joblib')
        print(f"Saving scaler to {filepath}")
        os.makedirs(save_scaler_dir_path, exist_ok=True)
        joblib.dump(scaler, filepath)

    return X_train_scaled, X_test_scaled

def plot_loss_function_epochs(
    loss_epochs: list[float],
    num_samples: int,
    batch_size: int,
    learning_rate: float,
    save_fig_dir: Optional[str] = None,
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_epochs,
        x=list(range(1, len(loss_epochs) + 1)),
        mode='lines+markers',
        name='Training Loss'
    ))
    fig.update_layout(
        title=f"Training Loss per Epoch<br><sup>Batch size: {batch_size}, Train samples: {num_samples}, Learning rate: {learning_rate}</sup><br>",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        template="plotly_white",
    )
    if save_fig_dir is not None:
        os.makedirs(save_fig_dir, exist_ok=True)
        fig.write_html(os.path.join(save_fig_dir, "training_loss.html"))
    return fig

def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
):
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.close(fig)
    return fig

def f1_objective(threshold, y_true, y_pred_proba):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return -f1_score(y_true, y_pred)

def find_best_threshold(y_true, y_pred_proba):
    result = minimize_scalar(f1_objective, bounds=(0, 1), method='bounded', args=(y_true, y_pred_proba))
    best_threshold = result.x
    best_f1 = -result.fun
    return best_threshold, best_f1  # Return threshold and best F1 score