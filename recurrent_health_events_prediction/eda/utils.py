from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from scipy.stats import norm, lognorm, gamma, weibull_min, t, kstest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.utils import resample
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from tqdm import tqdm  # For progress bars (optional)
from scipy.stats import spearmanr

from recurrent_health_events_prediction.model.model_types import DistributionType

## Both Datasets

def compare_distributions(data: pd.Series, distributions: list[DistributionType], var_name: str, plot_xrange: Optional[tuple] = None) -> pd.DataFrame:
    # Mapping enum to scipy.stats distribution
    dist_map = {
        DistributionType.NORMAL: norm,
        DistributionType.LOG_NORMAL: lognorm,
        DistributionType.GAMMA: gamma,
        DistributionType.WEIBULL: weibull_min,
        DistributionType.STUDENT_T: t
    }
    results = {}
    data = data.dropna().values  # remove NaNs, use as numpy array

    for dist_type in distributions:
        dist = dist_map[dist_type]
        params = dist.fit(data)
        loglik = np.sum(dist.logpdf(data, *params))
        k = len(params)
        n = len(data)
        aic = 2*k - 2*loglik
        bic = k*np.log(n) - 2*loglik
        ks_stat, ks_p = kstest(data, dist.cdf, args=params)
        results[dist_type.value] = {
            'params': params,
            'loglik': loglik,
            'aic': aic,
            'bic': bic,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
        }
    
    # Print ranked results
    print(f"--- Fit Comparison for {var_name} ---")
    res_table = pd.DataFrame(results).T
    res_table = res_table.sort_values("aic")

    # Plot
    plt.figure(figsize=(10,6))
    sns.kdeplot(data, label='Empirical', fill=True, color='black', lw=2, alpha=0.25)
    x = np.linspace(np.min(data), np.max(data), 1000)
    for dist_type in distributions:
        dist = dist_map[dist_type]
        params = results[dist_type.value]['params']
        y = dist.pdf(x, *params)
        plt.plot(x, y, label=dist_type.value.capitalize())
    plt.title(f"Distribution Fits for {var_name}")
    plt.xlabel(var_name)
    if plot_xrange:
        plt.xlim(plot_xrange)
    plt.ylabel("Density")
    plt.legend()
    sns.despine()
    plt.show()
    
    return res_table[["aic", "bic", "loglik", "ks_stat", "ks_pvalue"]]

def plot_spearman_vs_target(
    df: pd.DataFrame,
    numerical_features: List[str],
    target: str,
    alpha: float = 0.05,
    order: str = "abs",   # "abs", "value", or "none"
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Compute Spearman correlations between numerical_features and target, plot them,
    and return (figure, result_df).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the features and target.
    numerical_features : List[str]
        Columns in df to correlate against target.
    target : str
        Target column in df.
    alpha : float, optional
        Significance threshold for p-values. Default is 0.05.
    order : {"abs","value","none"}, optional
        Sort bars by absolute correlation ("abs"), raw value descending ("value"),
        or leave in given order ("none"). Default "abs".
    figsize : (int, int), optional
        Figure size for the bar plot. Default (10, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure.
    result_df : pd.DataFrame
        Dataframe with columns: feature, spearman_corr, p_value, significant.
    """
    # Basic validation
    missing = [c for c in [target, *numerical_features] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in df: {missing}")

    correlations = []
    p_values = []

    for feat in numerical_features:
        coef, pval = spearmanr(df[feat], df[target], nan_policy='omit')
        correlations.append(coef)
        p_values.append(pval)

    result_df = pd.DataFrame({
        'feature': numerical_features,
        'spearman_corr': correlations,
        'p_value': p_values
    })

    # Mark significance
    result_df['significant'] = result_df['p_value'] < alpha

    # Ordering
    if order == "abs":
        result_df = result_df.reindex(
            result_df['spearman_corr'].abs().sort_values(ascending=False).index
        )
    elif order == "value":
        result_df = result_df.sort_values('spearman_corr', ascending=False)
    elif order == "none":
        pass
    else:
        raise ValueError("order must be one of {'abs','value','none'}")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=result_df,
        x='feature',
        y='spearman_corr',
        hue='significant',
        palette={True: 'tab:blue', False: 'tab:gray'},
        ax=ax
    )
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title(f"Spearman Correlation vs. {target}\n(Blue = Significant, Gray = Not Significant)")
    ax.set_ylabel("Spearman Correlation")
    ax.set_xlabel("Feature")
    ax.legend(title='Significant')
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')
    fig.tight_layout()

    return fig, result_df

def plot_mutual_information(df, title='Mutual Information with Target'):
    # Sort for readability
    df_sorted = df.sort_values('MI_Median', ascending=True).reset_index(drop=True)

    # Extract values
    features = df_sorted['Feature']
    mi_median = df_sorted['MI_Median']
    ci_lower = df_sorted['CI_Lower']
    ci_upper = df_sorted['CI_Upper']

    # Calculate asymmetric error bars (distance from median)
    error_lower = mi_median - ci_lower
    error_upper = ci_upper - mi_median
    error = [error_lower, error_upper]

    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # Create bar plot with asymmetric CI error bars
    plt.barh(
        y=features,
        width=mi_median,
        xerr=error,
        color='skyblue',
        capsize=5
    )

    plt.xlabel('Mutual Information (Median ± 95% CI, in nats)')
    plt.title(title, pad=20, fontsize=14)
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def get_mutual_info_scores(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    subject_id_col: str,
    regression: bool = True,
    n_bootstraps: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Calculate Mutual Information (MI) scores with block bootstrap for clustered data.
    
    Args:
        df: Input DataFrame.
        features: List of feature columns to evaluate.
        target_col: Target variable column.
        subject_id_col: Column name identifying subjects/clusters.
        regression: If True, uses mutual_info_regression; else mutual_info_classif.
        n_bootstraps: Number of bootstrap iterations.
        random_state: Random seed for reproducibility.
        n_jobs: Number of CPU cores to use (-1 = all).
        
    Returns:
        DataFrame with MI scores (mean ± 95% CI) across bootstraps.
    """
    np.random.seed(random_state)
    subject_ids = df[subject_id_col].unique()
    results = {feat: [] for feat in features}
    
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Resample subjects with replacement
        boot_subjects = resample(subject_ids, replace=True)
        boot_df = df[df[subject_id_col].isin(boot_subjects)]
        
        # Compute MI for this bootstrap sample
        X = boot_df[features]
        y = boot_df[target_col]
        if regression:
            mi = mutual_info_regression(X, y, random_state=random_state, n_jobs=n_jobs)
        else:
            mi = mutual_info_classif(X, y, random_state=random_state, n_jobs=n_jobs)
        
        for feat, score in zip(features, mi):
            results[feat].append(score)
    
    # Aggregate results
    mi_stats = []
    for feat in features:
        scores = np.array(results[feat])
        mi_stats.append({
            'Feature': feat,
            'MI_Mean': np.mean(scores),
            'MI_Median': np.median(scores),
            'MI_Std': np.std(scores),
            'CI_Lower': np.percentile(scores, 2.5),
            'CI_Upper': np.percentile(scores, 97.5),
        })
    
    return pd.DataFrame(mi_stats).sort_values('MI_Median', ascending=False)

def get_rf_importance_scores(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    subject_id_col: str,
    task_type: str = "regression",  # "regression" or "classification"
    importance_type: str = "gini",  # "gini" or "permutation"
    n_bootstraps: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute Random Forest feature importance with block bootstrap for clustered data.
    Supports both regression and classification tasks.

    Args:
        df: Input DataFrame
        features: List of feature columns
        target_col: Target variable column
        subject_id_col: Column identifying subjects/clusters
        task_type: "regression" or "classification"
        importance_type: "gini" (mean decrease impurity) or "permutation"
        n_bootstraps: Number of bootstrap iterations
        random_state: Random seed
        n_jobs: CPU cores to use (-1 = all)
        verbose: Show progress bar

    Returns:
        DataFrame with importance scores and confidence intervals
    """
    # Validate inputs
    assert task_type in ["regression", "classification"], "task_type must be 'regression' or 'classification'"
    assert importance_type in ["gini", "permutation"], "importance_type must be 'gini' or 'permutation'"
    
    np.random.seed(random_state)
    subject_ids = df[subject_id_col].unique()
    results = {feat: [] for feat in features}
    
    # Initialize progress bar
    iter_range = tqdm(range(n_bootstraps), desc="Bootstrapping") if verbose else range(n_bootstraps)
    
    for _ in iter_range:
        # Block bootstrap: resample subjects with replacement
        boot_subjects = resample(subject_ids, replace=True)
        boot_df = df[df[subject_id_col].isin(boot_subjects)]
        
        # Initialize model
        if task_type == "regression":
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=n_jobs
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight="balanced"  # Handle imbalanced classes
            )
        
        model.fit(boot_df[features], boot_df[target_col])
        
        # Calculate importance scores
        if importance_type == "gini":
            scores = model.feature_importances_
        else:  # permutation importance
            perm_result = permutation_importance(
                model,
                boot_df[features],
                boot_df[target_col],
                n_repeats=5,
                random_state=random_state,
                n_jobs=n_jobs
            )
            scores = perm_result.importances_mean
        
        # Store results
        for feat, score in zip(features, scores):
            results[feat].append(score)
    
    # Aggregate results
    importance_stats = []
    for feat in features:
        feat_scores = np.array(results[feat])
        importance_stats.append({
            "feature": feat,
            "importance_mean": np.mean(feat_scores),
            "importance_median": np.median(feat_scores),
            "importance_std": np.std(feat_scores),
            "ci_lower": np.percentile(feat_scores, 2.5),
            "ci_upper": np.percentile(feat_scores, 97.5),
        })
    
    result_df = pd.DataFrame(importance_stats).sort_values("importance_median", ascending=False)
    return result_df

## MIMIC-III Dataset

def get_disease_recurrence(disease_recurrence_df: pd.DataFrame, disease: str, col_name: str, min_visits: int) -> pd.DataFrame:
    """
    For a given disease, find all patients ever diagnosed, count their hospital visits, 
    and plot the visit distribution for patients with more than min_visits.

    Args:
        disease_recurrence_df: DataFrame with at least SUBJECT_ID, HADM_ID, COMORBIDITY columns.
        disease: Name of the disease to look for (must match COMORBIDITY column).
        col_name: Column name for the resulting visit counts.
        min_visits: Minimum number of visits to include in the histogram.

    Returns:
        visits_df: DataFrame with SUBJECT_ID and visit count.
    """
    patients_with_disease_df = disease_recurrence_df[disease_recurrence_df['COMORBIDITY'] == disease]
    subjects_ids = patients_with_disease_df['SUBJECT_ID'].unique()

    if len(subjects_ids) == 0:
        raise ValueError(f"No patients found for disease: {disease}")

    visits_df = (
        disease_recurrence_df[disease_recurrence_df['SUBJECT_ID'].isin(subjects_ids)]
        .groupby('SUBJECT_ID')
        .agg({"HADM_ID": "nunique"})
        .reset_index()
        .rename(columns={"HADM_ID": col_name})
    )

    # Histogram for visualization (patients with more than min_visits)
    filtered = visits_df[visits_df[col_name] > min_visits]
    plt.figure(figsize=(10, 6))
    ax = filtered[col_name].plot.hist(
        bins=range(filtered[col_name].min(), filtered[col_name].max() + 2),
        edgecolor='black',
        alpha=0.7
    )
    ax.set_title(f'Distribution of {col_name} (Visits > {min_visits})', fontsize=14)
    ax.set_xlabel(col_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    if filtered[col_name].max() < 30:  # For big numbers, don't clutter
        ax.set_xticks(range(filtered[col_name].min(), filtered[col_name].max() + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return visits_df

def load_admissions_specific_disease(disease: str):
    admissions_df = pd.read_csv('../mimic-iii-dataset/ADMISSIONS.csv')
    diagnoses_icd_df = pd.read_csv('../mimic-iii-dataset/DIAGNOSES_ICD.csv')
    diagnoses_codes_df = pd.read_csv('../mimic-iii-dataset/D_ICD_DIAGNOSES.csv')

    disease_recurrence_df = pd.merge(diagnoses_icd_df, admissions_df[['HADM_ID', 'ADMITTIME', 'DISCHTIME']], on='HADM_ID', how='inner')
    disease_recurrence_df = pd.merge(disease_recurrence_df, diagnoses_codes_df[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']], on='ICD9_CODE', how='inner')

    with open('../config/diseases_codes.json', 'r') as f:
        diseases_codes = json.load(f)

    codes = diseases_codes.get(disease, [])
    if len(codes) == 0:
        raise ValueError(f"No codes found for disease: {disease}")
    
    
    subjects_id_s = disease_recurrence_df[disease_recurrence_df['ICD9_CODE'].isin(codes)]["SUBJECT_ID"].unique()

    disease_recurrence_df = disease_recurrence_df[disease_recurrence_df['SUBJECT_ID'].isin(subjects_id_s)]

    disease_recurrence_df['ADMITTIME'] = pd.to_datetime(disease_recurrence_df['ADMITTIME'])
    disease_recurrence_df['DISCHTIME'] = pd.to_datetime(disease_recurrence_df['DISCHTIME'])
    disease_recurrence_df['TYPE_DISEASE'] = np.where(disease_recurrence_df['ICD9_CODE'].isin(codes), disease, 'other')
    
    return disease_recurrence_df

def get_icu_admissions(disease_recurrence_df: pd.DataFrame, disease: str) -> pd.DataFrame:
    icu_admissions_df = pd.read_csv('../mimic-iii-dataset/ICUSTAYS.csv')
    icu_admissions_df = icu_admissions_df[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME']]
    
    disease_recurrence_df = pd.merge(disease_recurrence_df, icu_admissions_df, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    disease_recurrence_df['INTIME'] = pd.to_datetime(disease_recurrence_df['INTIME'])
    disease_recurrence_df['OUTTIME'] = pd.to_datetime(disease_recurrence_df['OUTTIME'])
    
    return disease_recurrence_df

def get_main_metrics_specific_cols(df: pd.DataFrame, cols: list, metrics: Optional[list[str]] = None) -> pd.DataFrame:
    metrics_dict = {}
    for metric in metrics:
        if metric == 'mean':
            metrics_dict['mean'] = df[cols].mean()
        elif metric == 'median':
            metrics_dict['median'] = df[cols].median()
        elif metric == 'mode':
            metrics_dict['mode'] = df[cols].mode().iloc[0]  # take the first mode
        elif metric == 'std':
            metrics_dict['std'] = df[cols].std()
        elif metric == 'skew':
            metrics_dict['skew'] = df[cols].skew()
        elif metric == 'kurtosis':
            metrics_dict['kurtosis'] = df[cols].kurtosis()
        elif metric == 'min':
            metrics_dict['min'] = df[cols].min()
        elif metric == 'max':
            metrics_dict['max'] = df[cols].max()

    metrics_df = pd.DataFrame(metrics_dict, index=cols).T
    metrics_df = metrics_df.round(3)  # Round to 3 decimal places for better readability
    metrics_df.index.name = 'Metric'
    metrics_df.reset_index(inplace=True)
    return metrics_df

## Drug Relapse Dataset

def plot_drug_history_of_a_donor(donor_id: str, drug_tests_df: pd.DataFrame):
    donor_df = drug_tests_df[drug_tests_df.donor_id == donor_id].copy()

    # Ensure that ScheduledDate is in datetime format
    donor_df['time'] = pd.to_datetime(donor_df['time'])

    # Sort the DataFrame by ScheduledDate
    donor_df = donor_df.sort_values(by='time')

    donor_df = donor_df.groupby(["donor_id", "drug_class", "time"]).agg({"drug_test_positive": "any"}).reset_index()

    # Create separate DataFrames for positive, negative, and NaN events
    positive_events = donor_df[donor_df['drug_test_positive'] == 1]
    negative_events = donor_df[donor_df['drug_test_positive'] == 0]
    nan_events = donor_df[donor_df['drug_test_positive'].isna()]

    # Create scatter plots for positive, negative, and NaN events
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=negative_events['time'],
        y=negative_events['drug_class'],
        mode='markers',
        marker=dict(color='darkgreen', size=12),
        name='Negative'
    ))

    fig.add_trace(go.Scatter(
        x=positive_events['time'],
        y=positive_events['drug_class'],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Positive'
    ))

    fig.add_trace(go.Scatter(
        x=nan_events['time'],
        y=nan_events['drug_class'],
        mode='markers',
        marker=dict(color='gray', size=12),
        name='NaN'
    ))

    # Update the layout to customize the font of the x and y axis titles
    fig.update_layout(
    title=f'Events for Donor {donor_id}',
    xaxis_title='Time',
    yaxis_title='Drug Class',
    showlegend=True,
    xaxis=dict(
        tickfont=dict(size=14),  # Increase the size of the x-axis labels
        title=dict(
            text='Time',
            font=dict(size=16, family='Arial', color='blue')  # Customize the font of the x-axis title
        )
    ),
    yaxis=dict(
        tickfont=dict(size=14),  # Increase the size of the y-axis labels
        title=dict(
            text='Drug Class',
            font=dict(size=16, family='Arial', color='blue')  # Customize the font of the y-axis title
        )
        )
    )

    # Show the plot
    fig.show()

def plot_positive_dates_timeline_of_a_donor(donor_id: str, drug_tests_df: pd.DataFrame):
    donor_df = drug_tests_df[drug_tests_df.donor_id == donor_id].copy()
    # Ensure 'time' column is in datetime format
    donor_df['time'] = pd.to_datetime(donor_df['time'])

    donor_df = donor_df.groupby("time").agg(positive_date=("drug_test_positive", "any")).reset_index()

    # Separate data based on result values
    positive_events = donor_df[donor_df['positive_date'] == 1]
    negative_events = donor_df[donor_df['positive_date'] == 0]

    # Create a plotly figure
    fig = go.Figure()

    # Add red dots for result = 0
    fig.add_trace(go.Scatter(
        x=negative_events['time'],
        y=[1] * len(negative_events),  # Use a constant y-value for timeline
        mode='markers',
        marker=dict(color='green', size=10),
        name='Negative'
    ))

    # Add green dots for result = 1
    fig.add_trace(go.Scatter(
        x=positive_events['time'],
        y=[1] * len(positive_events),  # Use a constant y-value for timeline
        mode='markers',
        marker=dict(color='red', size=10),
        name='Positive'
    ))

    # Update layout
    fig.update_layout(
        title='Timeline of Results',
        xaxis_title='Time',
        yaxis_title='',
        yaxis=dict(showticklabels=False),  # Hide y-axis labels
        showlegend=True
    )

    # Show the plot
    fig.show()

def get_drug_test_overall_stats_per_donor(drug_tests_df: pd.DataFrame):
    # Get the overall statistics for each donor
    drug_tests_df = drug_tests_df.groupby(by=["donor_id", "time"]).agg({
        'drug_test_positive': 'any',
    }).reset_index()

    overall_stats_df = drug_tests_df.groupby('donor_id').agg({
        'drug_test_positive': ['mean', 'sum', 'count'],
        'time': ['min', 'max'],
        'showedup': 'mean'
    }).reset_index()
    overall_stats_df.columns = ['SUBJECT_ID', 'POSITIVE_RATE', 'NUM_POSITIVE_DAYS', 'NUM_TEST_DAYS', 'FIRST_TEST_DATE', 'LAST_TEST_DATE', 'MEAN_SHOWED_UP']

    overall_stats_df['FIRST_TEST_DATE'] = pd.to_datetime(overall_stats_df['FIRST_TEST_DATE'])
    overall_stats_df['LAST_TEST_DATE'] = pd.to_datetime(overall_stats_df['LAST_TEST_DATE'])
    overall_stats_df['PARTICIPATION_DAYS'] = (overall_stats_df['LAST_TEST_DATE'] - overall_stats_df['FIRST_TEST_DATE']).dt.days

    return overall_stats_dfx