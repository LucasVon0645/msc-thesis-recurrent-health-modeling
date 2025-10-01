import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

from recurrent_health_events_prediction.data_extraction.data_types import DrugClass, ProgramType

def select_subjects_with_at_least_n_events(df: pd.DataFrame, n: int, subject_id_col: str) -> pd.DataFrame:
    """
    Select subjects with at least n events.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subject_id and one row per event.
    n : int
        Minimum number of events required for a subject to be included.
    -------
    pd.DataFrame
        DataFrame containing only subjects with at least n events.
    """
    return df.groupby(subject_id_col).filter(lambda x: len(x) >= n)

def get_past_events(df, subject_id_col):
    """
    Get all past events for each subject, excluding the last event.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subject_id and one row per event.
    subject_id_col : str
        Column name identifying the subject.
    -------
    pd.DataFrame
        DataFrame containing all past events for each subject, excluding the last event.
    """
    return df.groupby(subject_id_col, group_keys=False).apply(lambda group: group.iloc[:-1])

def get_last_event(df, subject_id_col):
    """
    Get the last event for each subject.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing subject_id and one row per event.
    subject_id_col : str
        Column name identifying the subject.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing only the last event for each subject.
    """
    return df.groupby(subject_id_col, group_keys=False).apply(lambda group: group.iloc[-1:])

def get_rows_up_to_event_id(
    df,
    event_id_col: str,
    event_ids: pd.Series,
    id_col: str = 'SUBJECT_ID',
    time_col: str = 'ADMITTIME',
    include_event_id: bool = True
) -> pd.DataFrame:
    """
    Truncate each subject's rows up to a given event_id, provided per subject.

    Parameters:
        df (pd.DataFrame): Full dataset.
        event_id_col (str): Column containing unique event identifiers.
        event_ids (pd.Series): Series mapping subject_id → event_id.
        id_col (str): Column identifying the subject.
        time_col (str): Column to sort within each group.
        include_event_id (bool): Whether to include the row with the event_id.

    Returns:
        pd.DataFrame: Truncated dataframe with rows up to each subject's event_id.
    """
    def truncate_group(group):
        subject_id = group[id_col].iloc[0]
        target_event_id = event_ids.get(subject_id, None)
        if target_event_id is None:
            return pd.DataFrame()  # skip if no target event for subject
        group = group.sort_values(by=time_col)
        target_event_id = target_event_id.item()
        if target_event_id not in group[event_id_col].values:
            return pd.DataFrame()  # skip if event_id not found
        idx = group[group[event_id_col] == target_event_id].index[-1]  # in case of duplicates
        if include_event_id:
            return group.loc[:idx]
        else:
            return group.loc[:idx].drop(index=idx)

    return df.groupby(id_col, group_keys=False).apply(truncate_group)

def bin_time_col_into_cat(df, bins, labels, cat_col_name, col_to_bin, apply_encoding: bool = True) -> tuple:
    """
    Bins a continuous time column into categorical bins and adds a new column to the DataFrame.
    Parameters:
    - df: DataFrame containing the time column to bin.
    - bins: List of bin edges to use for binning.
    - labels: List of labels for the bins.
    - cat_col_name: Name of the new categorical column to create.
    - col_to_bin: Name of the column to bin.
    Returns:
    - df: DataFrame with the new categorical column added.
    - event_time_cat_mapping: Dictionary mapping the categorical codes to their labels.
    """
    if bins[-1] != np.inf:
        bins = bins + [np.inf]
    df[cat_col_name] = pd.cut(
        df[col_to_bin],
        bins=bins,
        include_lowest=True,
        right=False,
        labels=labels)

    event_time_cat_mapping = dict({idx: label for idx, label in enumerate(df[cat_col_name].cat.categories)})
    if apply_encoding:
        df[cat_col_name] = df[cat_col_name].cat.codes

    return df, event_time_cat_mapping

def build_sequences_up_to_last_events(last_events_df, all_events_df, time_col='EVENT_DURATION', time_col_cat='READMISSION_TIME_CAT',
                                      time_labels = None, time_bins = None, include_event_id=True):
    """
    Builds sequences of events up to the last event for each subject, binning the time column into categories.
    Parameters:
    - last_events_df: DataFrame containing the last event for each subject.
    - all_events_df: DataFrame containing all events for all subjects.
    - time_col: Name of the column containing the time duration.
    - time_col_cat: Name of the new categorical column to create for binned time.
    - time_labels: List of labels for the time bins. If None, default labels will be used.
    - time_bins: List of bin edges for the time column. If None, default bins will be used.
    - include_event_id: Whether to include the event_id in the output sequences.
    Returns:
    - events_up_to_the_last_one_df: DataFrame containing sequences of events up to the last event for each subject.
    - time_cat_mapping: Dictionary mapping the categorical codes to their labels.
    """
    if time_col not in last_events_df.columns or time_col not in all_events_df.columns:
        raise ValueError(f"Time column '{time_col}' must be present in both last_events_df and all_events_df.")
    
    if time_bins is None or time_labels is None:
        time_bins = [0, 30, 120]  # Default bin edges in days
        labels = ['0-30', '30-120', '120+']

    time_bins = time_bins + [np.inf]  # Add infinity to the last bin to include all values greater than the last bin edge

    print(f"Labels for {time_col_cat}: {labels}")
    print(f"Bins for {time_col_cat}: {time_bins}")

    last_events_df, time_cat_mapping = bin_time_col_into_cat(last_events_df, time_bins, labels, time_col_cat, time_col, apply_encoding=False)
    all_events_df, _ = bin_time_col_into_cat(all_events_df, time_bins, labels, time_col_cat, time_col, apply_encoding=False)
    last_events_ids = last_events_df[["SUBJECT_ID", "HADM_ID"]].set_index("SUBJECT_ID")["HADM_ID"]
    events_up_to_the_last_one_df = get_rows_up_to_event_id(all_events_df, 'HADM_ID', last_events_ids, include_event_id=include_event_id)

    return events_up_to_the_last_one_df, time_cat_mapping

def calculate_past_rolling_stats(
    df: pd.DataFrame,
    group_col: str,
    feature: str,
    stats: list = None,
    prefix: str = None,
    id_col: str = None    # NEW PARAMETER
) -> pd.DataFrame:
    """
    For each row, computes rolling stats over all previous values of `feature`
    within each `group_col`. Does not include current row in stat calculation.
    Returns a DataFrame with the rolling statistics as columns.
    If id_col is provided, it will be the first column in the result.
    """
    if stats is None:
        stats = ['mean', 'sum', 'median', 'count', 'std']
    if prefix is None:
        prefix = feature

    # Helper function to compute all requested stats
    def calc_stats(x):
        x_shifted = x.shift()
        res = {}
        if 'mean' in stats:
            res[f'{prefix}_PAST_MEAN'] = x_shifted.expanding().mean()
        if 'sum' in stats:
            res[f'{prefix}_PAST_SUM'] = x_shifted.expanding().sum()
        if 'median' in stats:
            res[f'{prefix}_PAST_MEDIAN'] = x_shifted.expanding().median()
        if 'count' in stats:
            res[f'{prefix}_PAST_COUNT'] = x_shifted.expanding().count()
        if 'std' in stats:
            res[f'{prefix}_PAST_STD'] = x_shifted.expanding().std()
        return pd.DataFrame(res)
    
    # Apply for each group
    stats_df = (
        df.groupby(group_col)[feature]
        .apply(calc_stats)
        .reset_index(level=0, drop=True)
    )
    # Now stats_df has the same index as df

    # Add group_col and optional id_col
    stats_df[group_col] = df[group_col].values
    if id_col is not None:
        stats_df[id_col] = df[id_col].values

    # Reorder columns: id_col (if present), group_col, then stat columns
    col_order = []
    if id_col is not None:
        col_order.append(id_col)
    col_order.append(group_col)
    stat_cols = [c for c in stats_df.columns if c not in col_order]
    stats_df = stats_df[col_order + stat_cols]

    return stats_df

def calculate_past_rolling_stats_multiple_features(
    df: pd.DataFrame,
    group_col: str,
    feature,
    stats: list = None,
    prefix = None,
    id_col: str = None  # New parameter
) -> pd.DataFrame:
    """
    For each row, computes rolling stats over all previous values of `feature`
    within each `group_col`. Does not include current row in stat calculation.
    Returns a DataFrame with the rolling statistics as columns, plus id_col and group_col.
    Supports both single feature string or list of features.
    Prefix must be None, a string (applied to all), or a list of strings of same length as features.
    """
    if stats is None:
        stats = ['mean', 'sum', 'median', 'count', 'std']
    
    # Normalize features
    if isinstance(feature, str):
        features = [feature]
    else:
        features = feature
    
    # Normalize prefix
    if prefix is None:
        prefixes = features
    elif isinstance(prefix, str):
        if len(features) > 1:
            raise ValueError("Features must be a single string if 'prefix' is a string.")
        prefixes = [prefix]
    else:
        if len(prefix) != len(features):
            raise ValueError("If 'prefix' is a list, it must be the same length as 'feature' list.")
        prefixes = prefix

    # Function to apply to each group
    def calc_stats(group):
        shifted = group[features].shift()
        results = pd.DataFrame(index=group.index)
        for f, p in zip(features, prefixes):
            for stat in stats:
                colname = f"{p}_PAST_{stat.upper()}"
                stat_func = getattr(shifted[f].expanding(), stat)
                results[colname] = stat_func()
        return results

    stats_df = (
        df.groupby(group_col, group_keys=False)
          .apply(calc_stats)
    )
    stats_df.index = df.index

    # Add id_col and group_col to results
    if id_col is not None:
        stats_df[id_col] = df[id_col].values
    stats_df[group_col] = df[group_col].values

    # Order columns: id_col, group_col, features
    cols = []
    if id_col is not None:
        cols.append(id_col)
    cols.append(group_col)
    # Add summary columns in their existing order
    summary_cols = [c for c in stats_df.columns if c not in cols]
    stats_df = stats_df[cols + summary_cols]

    return stats_df

def hot_encode_drug_classes(df, col):
    df[col] = df[col].apply(ast.literal_eval)

    # Fit the MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    hot_encoded = mlb.fit_transform(df[col])

    # Create new column names in UPPER case and with _POS
    new_cols = [drug.upper() + '_POS' for drug in mlb.classes_]

    # Create a DataFrame for the encoded variables
    hot_encoded_df = pd.DataFrame(hot_encoded, columns=new_cols, index=df.index)

    # Concatenate with original DataFrame
    return pd.concat([df, hot_encoded_df], axis=1)

def filter_select_only_one_program_type(df: pd.DataFrame, program_type_col: str, program_type: ProgramType) -> pd.DataFrame:
    """
    Filter the DataFrame to select only rows with a specific program type.
    
    Parameters:
    - df: DataFrame containing the program type column.
    - program_type_col: Name of the column containing program types.
    - program_type: The specific program type to filter by.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the specified program type.
    """
    program_type_str = program_type.value if isinstance(program_type, ProgramType) else program_type
    return df[df[program_type_col] == program_type_str]

def filter_select_only_one_drug_class(df: pd.DataFrame, drug_class_col: str, drug_class: DrugClass) -> pd.DataFrame:
    """
    Filter the DataFrame to select only rows with a specific drug class.
    
    Parameters:
    - df: DataFrame containing the drug class column.
    - drug_class_col: Name of the column containing drug classes as strings (convertible to lists).
    - drug_class: The specific drug class to filter by.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the rows where the specified drug class is present.
    """
    drug_class_str = drug_class.value if isinstance(drug_class, DrugClass) else drug_class
    df[drug_class_col] = df[drug_class_col].apply(ast.literal_eval)
    return df[df[drug_class_col].apply(lambda x: drug_class_str in x)]

def map_mimic_races(race):
    if 'BLACK' in race or 'AFRICAN' in race:
        return 'BLACK'
    elif 'HISPANIC' in race or 'LATINO' in race:
        return 'HISPANIC'
    elif race == 'WHITE':
        return 'WHITE'
    else:
        return 'OTHER'
    
def remap_mimic_races(df):
    """Remap the 'ETHNICITY' column in the DataFrame to a simplified set of categories."""
    df['ETHNICITY'] = df['ETHNICITY'].apply(map_mimic_races)
    return df

def remap_discharge_location(df):
    """
    Simplify DISCHARGE_LOCATION into 3 categories:
    HOME, POST_ACUTE_CARE, OTHERS
    """
    # Internal mapping
    discharge_category_map = {
        # HOME group
        "HOME": "HOME",
        "HOME HEALTH CARE": "HOME",
        "HOME WITH HOME IV PROVIDR": "HOME",

        # Post-acute care
        "SNF": "POST_ACUTE_CARE",
        "REHAB/DISTINCT PART HOSP": "POST_ACUTE_CARE",
        "LONG TERM CARE HOSPITAL": "POST_ACUTE_CARE",
        "HOSPICE-HOME": "POST_ACUTE_CARE",
        "HOSPICE-MEDICAL FACILITY": "POST_ACUTE_CARE",
        "ICF": "POST_ACUTE_CARE",  # Intermediate Care Facility

        # Others
        "DEAD/EXPIRED": "OTHERS",
        "DISC-TRAN CANCER/CHLDRN H": "OTHERS",
        "SHORT TERM HOSPITAL": "OTHERS",
        "LEFT AGAINST MEDICAL ADVI": "OTHERS",
        "DISCH-TRAN TO PSYCH HOSP": "OTHERS",
        "OTHER FACILITY": "OTHERS",
        "DISC-TRAN TO FEDERAL HC": "OTHERS"
    }
    df['DISCHARGE_LOCATION'] = df['DISCHARGE_LOCATION'].astype(str).str.upper().str.strip()  # Ensure uppercase for consistency
    # Apply mapping with default to OTHERS
    df["DISCHARGE_LOCATION"] = df["DISCHARGE_LOCATION"].map(
        lambda x: discharge_category_map.get(x, "OTHERS")
    )
    
    return df
