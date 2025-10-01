from typing import Optional
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from recurrent_health_events_prediction.preprocessing.utils import get_rows_up_to_event_id
from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import RecurrentHealthEventsHMM


def set_observation_window(row, limit=120, event_col="READMISSION_EVENT"):
    if row['EVENT_DURATION'] > limit:
        row['EVENT_DURATION'] = limit
        row[event_col] = 0
    if row[event_col] == 0:
        row['EVENT_DURATION'] = limit
    return row

def filter_last_events_mimic(df, select_only_first_event=True, exclude_elective=True, limit=120):
    if select_only_first_event:
        df = df.sort_values(['SUBJECT_ID', 'ADMITTIME']).groupby('SUBJECT_ID', as_index=False).first()
    if exclude_elective:
        df = df[df["NEXT_ADMISSION_TYPE"] != "ELECTIVE"]
    df = df.apply(lambda r: set_observation_window(r, limit, event_col="READMISSION_EVENT"), axis=1)
    return df

def apply_set_observation_window_mimic(df, limit=120):
    return df.apply(lambda r: set_observation_window(r, limit), axis=1)

def hide_time_to_next_event_hmm_feature(df: pd.DataFrame, event_id_col: str, event_ids: list, time_feature_col: str):
    events_to_hide = df[event_id_col].isin(event_ids)
    df[time_feature_col] = np.where(events_to_hide, np.nan, df[time_feature_col])
    return df

def get_events_up_to_event(all_events_df, event_ids, event_id_col="HADM_ID", time_col = "ADMITTIME", id_col="SUBJECT_ID"):
    return get_rows_up_to_event_id(all_events_df, event_id_col, event_ids, include_event_id=True, time_col=time_col, id_col=id_col)

def save_df(df, path):
    df.to_csv(path, index=False)

def compute_hidden_state_probs(hmm_model: RecurrentHealthEventsHMM, events_df):
    return hmm_model.predict_proba_last_obs_partial(events_df)

def build_hidden_state_results(
    last_event_ids: list,
    hidden_state_probs: list,
    idx_to_labels: dict,
    labels_to_idx: dict,
    event_id_col="HADM_ID",
):
    hidden_state_probs = np.array(hidden_state_probs)
    hidden_states = np.argmax(hidden_state_probs, axis=1)
    hidden_states_str = list(map(lambda x: idx_to_labels[x], hidden_states))
    result_dict = {
        event_id_col: last_event_ids,
        "HEALTH_HIDDEN_RISK": hidden_states_str,
    }
    for label in idx_to_labels.values():
        idx = labels_to_idx[label]
        probs_of_hidden_state = hidden_state_probs[:, idx]
        probs_of_hidden_state = np.where(
            probs_of_hidden_state < 1e-2, 0, probs_of_hidden_state
        )
        result_dict[f"PROB_HIDDEN_RISK_{label.upper()}"] = probs_of_hidden_state
    return pd.DataFrame(result_dict)

def run_chi2_and_plot(last_events_df, save_dir: Optional[str] = None, event_time_cat_col='READMISSION_TIME_CAT', show_plot: bool = False):
    contingency_table = pd.crosstab(last_events_df['HEALTH_HIDDEN_RISK'], last_events_df[event_time_cat_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    test_results_string = (
        f"Chi-squared statistic: {chi2}\n"
        f"P-value: {p}\n"
        f"Degrees of freedom: {dof}\n"
    )
    if save_dir:
        test_results_path = os.path.join(save_dir, "chi_squared_test_results.txt")
        print(f"Chi-squared test results saved to {test_results_path}")
        with open(test_results_path, "w") as f:
            f.write(test_results_string)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Contingency Table: Health Hidden Risk vs Event Time Category")
    ax.set_xlabel("Event Time Category")
    ax.set_ylabel("Health Hidden Risk")
    if save_dir:
        plot_path = os.path.join(save_dir, "target_health_hidden_risk_vs_event_time_cat.png")
        fig.savefig(plot_path)
    if show_plot:
        plt.show()
    plt.close()
    
    return test_results_string, fig

def plot_facet_hidden_state_distribution(
    df,
    save_path=None,
    hue_order=None,
    axis_order=None,
    event_name="readmission",
    figsize=(12, 8),
    show_plot=False,
    title = "Histogram of Health Hidden Risk by Event"
    ):
    """
    Faceted histogram of HEALTH_HIDDEN_RISK by event_name using seaborn.
    - df: DataFrame containing 'HEALTH_HIDDEN_RISK' and 'READMISSION_EVENT' columns
    - save_path: Where to save the plot (PNG file)
    - hue_order: Order of health hidden risk categories (list of label names, not integers)
    - figsize: tuple for size
    """
    event_col = f"{event_name.upper()}_EVENT"
        # --- Set category order for HEALTH_HIDDEN_RISK ---
    if axis_order is not None:
        df = df.copy()  # avoid SettingWithCopyWarning
        df["HEALTH_HIDDEN_RISK"] = pd.Categorical(
            df["HEALTH_HIDDEN_RISK"],
            categories=axis_order,
            ordered=True
        )

    g = sns.displot(
        df,
        x="HEALTH_HIDDEN_RISK",
        col=event_col,
        hue="HEALTH_HIDDEN_RISK",
        multiple="stack",
        col_wrap=2,
        facet_kws={'sharey': False},
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        hue_order=hue_order,
        shrink=0.9
    )
    g.set_axis_labels("Health Hidden Risk", "Count")
    g.set_titles(col_template="{col_name} ({col_var})")
    fig = g.figure
    fig.suptitle(title)
    if save_path:
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(fig)
    print(f"Facet histogram plot saved to {save_path}")
    return fig

def plot_event_duration_kde_by_hidden_state(
    df: pd.DataFrame,
    event_name: str,
    hidden_state_label_order: list,
    save_path: str
) -> str:
    """
    Plots and saves a KDE of event duration by hidden health risk state for readmitted events.

    Args:
        df (pd.DataFrame): DataFrame containing event data and HMM features.
        event_name (str): Name of the event (used to find event column and label plot).
        hidden_state_label_order (list): Order of hidden state labels for consistent coloring.
        save_dir (str): Directory to save the resulting plot.

    Returns:
        str: Path to the saved plot image.
    """
    event_col = f"{event_name.upper()}_EVENT"

    # Filter only readmitted events
    filtered_df = df[df[event_col] == 1]

    # Calculate upper limit for clipping the KDE
    upper_limit = np.ceil(filtered_df['EVENT_DURATION'].max()).item()

    # Plot KDE
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(
        filtered_df,
        x='EVENT_DURATION',
        hue='HEALTH_HIDDEN_RISK',
        common_norm=False,
        hue_order=hidden_state_label_order,
        clip=(0, upper_limit),
        ax=ax
    )
    plt.title(f'Distribution of {event_name.title()} Duration (< {int(upper_limit)} days) by Pred. Health Hidden Risk')
    plt.xlabel('Event Duration (days)')
    plt.ylabel('Density')

    # Save plot
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return fig
