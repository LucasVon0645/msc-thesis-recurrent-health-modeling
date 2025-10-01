import os
import yaml
import numpy as np
import pandas as pd
from scipy.stats import poisson

from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import RecurrentHealthEventsHMM, get_model_selection_results_hmm
from recurrent_health_events_prediction.model.model_types import DistributionType
from recurrent_health_events_prediction.preprocessing.utils import hot_encode_drug_classes, remap_discharge_location
from recurrent_health_events_prediction.training.utils import preprocess_features_to_one_hot_encode
from recurrent_health_events_prediction.utils.neptune_utils import add_plot_to_neptune_run, upload_hmm_output_file_to_neptune, upload_model_to_neptune

import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional

def add_pred_prob_states_to_df(df, pred_proba, col_prefix="PROB_HIDDEN_RISK_", labels=None):
    """
    Adds the predicted state probabilities to the dataframe.
    """
    
    n_states = len(pred_proba[0][0])

    if labels is None:
        prob_cols = [f"{col_prefix}{i}" for i in range(n_states)]
    else:
        prob_cols = [f"{col_prefix}{labels[i].upper()}" for i in range(n_states)]

    flat_probs_list = [obs for seq in pred_proba for obs in seq]
    flat_probs_df = pd.DataFrame(flat_probs_list, columns=prob_cols)
    df = pd.concat([df.reset_index(drop=True), flat_probs_df], axis=1)

    return df

def add_pred_state_to_df(df, pred_seq, col_name="HEALTH_HIDDEN_RISK", labels_dict=None):
    """
    Adds the predicted hidden state to the dataframe.
    """
    
    if labels_dict is None:
        df[col_name] = [state for seq in pred_seq for state in seq]
    else:
        df[col_name] = [labels_dict[state] for seq in pred_seq for state in seq]

    return df

def create_and_save_params_df(hmm, labels=None, save_path=None, include_state_labels=True, neptune_run=None, neptune_path="hmm_params"):
    distributions_params_df = hmm.get_features_dist_df(include_state_labels)
    if labels is None:
        labels = hmm.config.get("hidden_state_labels")
    if 'State Label' in distributions_params_df.columns:
        distributions_params_df = distributions_params_df.sort_values(
            by='State Label', key=lambda col: [labels.index(label) for label in col]
        )
    # Round float columns
    for col in distributions_params_df.columns:
        distributions_params_df[col] = distributions_params_df[col].apply(lambda x: round(x, 3) if isinstance(x, float) else x)

    styled_df = (
        distributions_params_df.style
        .set_table_styles(
            [
                # Table
                {'selector': 'table', 'props': [('border-collapse', 'collapse')]},
                # Header cells (all levels)
                {'selector': 'th', 'props': [
                    ('font-size', '12px'),
                    ('text-align', 'center'),
                    ('border', '1px solid black'),
                    ('padding', '5px')
                ]},
                # Body cells
                {'selector': 'td', 'props': [
                    ('font-size', '12px'),
                    ('text-align', 'center'),
                    ('border', '1px solid black'),
                    ('padding', '5px')
                ]}
            ]
        )
    )

    # Save HTML file
    if save_path:
        with open(save_path, 'w') as f:
            f.write(styled_df.to_html())

    if neptune_run is not None:
        neptune_run[neptune_path].upload(save_path)

    return styled_df

def plot_feature_distribution_per_hidden_state(feature_name, distribution_params_df,
                                               show_plot: bool = True,
                                               hue_order=None, save_file=None):
    data = {}
    is_poisson = False

    for state_label in distribution_params_df['State Label'].unique():
        state_mask = (distribution_params_df['State Label'] == state_label)
        if (feature_name, 'Mean') in distribution_params_df.columns and (feature_name, 'Cov') in distribution_params_df.columns:
            mean = distribution_params_df.loc[state_mask, (feature_name, 'Mean')].values[0]
            variance = distribution_params_df.loc[state_mask, (feature_name, 'Cov')].values[0]
            scale = np.sqrt(variance)
            samples = np.random.normal(loc=mean, scale=scale, size=15000)
            data[str(state_label)] = samples
        elif (feature_name, 'Lambda') in distribution_params_df.columns:
            is_poisson = True
            lam = distribution_params_df.loc[state_mask, (feature_name, 'Lambda')].values[0]
            # Instead of sampling, plot the PMF over a reasonable k range
            k = np.arange(0, max(15, int(lam*3)))
            pmf = poisson.pmf(k, lam)
            data[str(state_label)] = (k, pmf)
        elif (feature_name, 'Shape') in distribution_params_df.columns and (feature_name, 'Rate') in distribution_params_df.columns:
            shape = distribution_params_df.loc[state_mask, (feature_name, 'Shape')].values[0]
            rate = distribution_params_df.loc[state_mask, (feature_name, 'Rate')].values[0]
            samples = np.random.gamma(shape, 1/rate, size=15000)
            data[str(state_label)] = samples
        else:
            print(f"Warning: No valid distribution parameters found for feature '{feature_name}' in state '{state_label}'.")
            continue  # Skip if no valid distribution parameters found
    
    if len(data) < 1:
        print(f"Warning: No data to plot for feature '{feature_name}'.")
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))

    if is_poisson:
        # Map each state label to a numeric index for bar offsets
        state_order = list(data.keys()) if hue_order is None else hue_order
        state_to_offset = {state: i for i, state in enumerate(state_order)}

        for state, (k, pmf) in data.items():
            offset = 0.1 * state_to_offset[state]
            ax.bar(k + offset, pmf, width=0.2, alpha=0.7, label=state)
      
        ax.set_xlabel(feature_name)
        ax.set_ylabel("PMF")
        ax.legend(title='Hidden State')
        plt.title(f"Predicted Dist. for {feature_name} by Hidden State")
        plt.tight_layout()
    else:
        # For continuous, still use KDE
        plot_df = pd.DataFrame({
            'value': np.concatenate(list(data.values())),
            'Hidden State': np.concatenate([[k]*len(v) for k, v in data.items()])
        })
        plot_df['Hidden State'] = plot_df['Hidden State'].astype(str)
        ax = sns.kdeplot(data=plot_df, x='value', hue='Hidden State', hue_order=hue_order, common_norm=False, ax=ax)
        plt.xlabel(feature_name)
        plt.ylabel("Density")
        plt.title(f"Predicted Dist. for {feature_name} by Hidden State")
        plt.tight_layout()

    if save_file:
        plt.savefig(save_file)
    if show_plot:
        plt.show()

    plt.close()
    return fig

def plot_cat_event_time_by_hidden_state(df, time_cat_col: str = 'READMISSION_TIME_CAT', hidden_state_col: str = 'HEALTH_HIDDEN_RISK',
                                              event_time_cat_mapping: Optional[dict] = None, show_plot: bool = True,
                                              hue_order=None,  # Optional hue order for the categories
                                              hidden_states_order=None, event_name="readmission",
                                              save_file: Optional[str] = None, title_suffix: str = ''):
    """
    Plots the distribution of readmission time by HMM hidden state.

    Parameters:
    - df: DataFrame containing the data.
    - readmission_time_cat_mapping: Optional dictionary mapping readmission time categories to labels.
    - time_cat_col: Column name for readmission time categories.
    - hidden_state_col: Column name for HMM hidden states.
    - show_plot: Whether to display the plot.
    - hue_order: Optional order for the hue categories.
    - hidden_states_order: Optional order for the hidden states.
    - save_file: Optional file path to save the plot.
    - title_suffix: Suffix to add to the plot title.
    """
    event_series = df[time_cat_col].map(event_time_cat_mapping) if event_time_cat_mapping else df[time_cat_col]
    # Cross-tabulate hidden state vs. event time
    ct = pd.crosstab(df[hidden_state_col], event_series)

    # Convert the crosstab to a long-form DataFrame for seaborn
    ct_long = ct.reset_index().melt(id_vars=hidden_state_col, var_name=time_cat_col, value_name='Count')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=ct_long, 
        x=hidden_state_col, 
        y='Count', 
        hue=time_cat_col, 
        hue_order=hue_order, 
        order=hidden_states_order, 
        ax=ax
    )
    
    title = f"Distribution of {event_name.title()} Time Category by HMM Hidden State" + title_suffix
    ax.set_title(title)
    ax.set_xlabel("HMM Hidden State")
    ax.set_ylabel("Count")
    ax.legend(title=f"{event_name.title()} Time Category")

    if save_file:
        fig.savefig(save_file)
        print(f"Plot saved to {save_file}")

    if show_plot:
        plt.show()

    plt.close()
    return fig

def plot_feature_kde_by_hidden_state(feature_col: str, df: pd.DataFrame,
                                     hidden_state_col: str = 'HEALTH_HIDDEN_RISK', show_plot: bool = True,
                                     hue_order=None, save_path: str = None, suffix_title: str = ''):
    fig, ax = plt.subplots(figsize=(10, 5))
    feature_name = feature_col.replace('_', ' ').title()
    sns.kdeplot(
        data=df,
        x=feature_col,
        hue=hidden_state_col,
        common_norm=False,
        hue_order=hue_order,
        ax=ax
    )
    title = f'Distribution of {feature_name} by {hidden_state_col.replace("_", " ").title()} {suffix_title}'
    plt.title(title)
    plt.xlabel(feature_name)
    plt.ylabel('Density')

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close()
    return fig

def plot_transition_matrix_heatmap(
    transition_matrix_df, 
    title="Transition Matrix Heatmap",
    cmap="Blues",
    show_plot=True,
    save_file=None
):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        transition_matrix_df,
        annot=True, fmt=".2f",
        cmap=cmap,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={"label": "Transition Probability"},
        ax=ax
    )
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title)
    plt.tight_layout()
    
    if save_file:
        fig.savefig(save_file, dpi=150)
    if show_plot:
        plt.show()
    plt.close()
    return fig

def load_and_prepare_historical_data_mimic(training_data_path: str, use_only_sequences_gt_2_steps=False):
    """
    Loads and prepares historical data for MIMIC dataset.

    Args:
        training_data_path (str): Path to the training data directory.
        use_only_sequences_gt_2_steps (bool, optional): If True, ignores sequences with length 1. Defaults to False.

    Returns:
        _type_: _description_
    """
    filepath_historical_events = os.path.join(training_data_path, "historical_events.csv")
    print("Loading data from: ", filepath_historical_events)
    historical_events_df = pd.read_csv(filepath_historical_events)
    historical_events_df["ADMITTIME"] = pd.to_datetime(historical_events_df["ADMITTIME"])
    historical_events_df["DISCHTIME"] = pd.to_datetime(historical_events_df["DISCHTIME"])
    historical_events_df["HAS_COPD"] = historical_events_df["HAS_COPD"].astype(int)
    historical_events_df["HAS_CONGESTIVE_HF"] = historical_events_df["HAS_CONGESTIVE_HF"].astype(int)
    median_log_days_since_last_hospitalization = historical_events_df['LOG_DAYS_SINCE_LAST_HOSPITALIZATION'].median()
    historical_events_df['LOG_DAYS_SINCE_LAST_HOSPITALIZATION'] = historical_events_df['LOG_DAYS_SINCE_LAST_HOSPITALIZATION'].fillna(median_log_days_since_last_hospitalization)

    historical_events_df = remap_discharge_location(historical_events_df)
    historical_events_df, new_cols = preprocess_features_to_one_hot_encode(
        historical_events_df,
        ["DISCHARGE_LOCATION"],
        one_hot_cols_to_drop=["DISCHARGE_LOCATION_OTHERS"],
    )
    historical_events_df[new_cols] = historical_events_df[new_cols].astype(int)


    print("Data loaded successfully. Preparing data...")

    # Filter data as in notebook
    mask = (historical_events_df["IN_HOSP_DEATH_EVENT"] == 0) & (historical_events_df["AFTER_HOSP_DEATH_EVENT"] == 0)
    historical_events_df = historical_events_df[mask]

    if use_only_sequences_gt_2_steps:
        print("Using only sequences with more than 2 steps.")
        # --- Multiple Readmissions Subset ---
        hospitalizations_count = historical_events_df.groupby('SUBJECT_ID').agg(
            HOSPITALIZATIONS_COUNT=('HADM_ID', 'nunique'),
            HOSPITALIZATIONS_WITH_READMISSION_COUNT=('READMISSION_EVENT', 'sum'),
        ).reset_index()
        subjects_with_multiple_readmissions_ids = hospitalizations_count[
            hospitalizations_count['HOSPITALIZATIONS_WITH_READMISSION_COUNT'] > 1
        ]['SUBJECT_ID'].unique()
        X = historical_events_df[historical_events_df["SUBJECT_ID"].isin(subjects_with_multiple_readmissions_ids)]
        X = X.sort_values(by=["SUBJECT_ID", "ADMITTIME"])
    else:
        print("Using all sequences, including those with 1 step.")
        X = historical_events_df
        X = X.sort_values(by=["SUBJECT_ID", "ADMITTIME"])
    return X, filepath_historical_events

def load_and_prepare_historical_data_relapse(training_data_path: str, use_only_sequences_gt_2_steps=False):
    """
    Loads and prepares historical data for relapse prediction.

    Args:
        training_data_path (str): path to the training data directory.
        use_only_sequences_gt_2_steps (bool, optional): If True, ignores sequences with length 1. Defaults to False.

    Returns:
        tuple: (DataFrame with historical events, file path of the historical events CSV)
    """
    # Load historical events
    filepath_historical_events = os.path.join(training_data_path, "historical_relapses.csv")
    print("Loading data from: ", filepath_historical_events)
    historical_events_df = pd.read_csv(filepath_historical_events)
    historical_events_df["RELAPSE_START"] = pd.to_datetime(historical_events_df["RELAPSE_START"])
    historical_events_df['PREV_RELAPSE_30_DAYS'] = historical_events_df['PREV_RELAPSE_30_DAYS'].astype(int)
    historical_events_df = hot_encode_drug_classes(historical_events_df, 'PREV_POSITIVE_DRUGS')

    print("Data loaded successfully. Preparing data...")

    if use_only_sequences_gt_2_steps:
        print("Using only sequences with more than 2 steps.")
        # --- Multiple Relapses Subset ---
        relapses_count = historical_events_df.groupby('DONOR_ID').agg(
            RELAPSES_COUNT=('COLLECTION_ID', 'nunique'),
        ).reset_index()
        subjects_with_multiple_relapses_ids = relapses_count[
            relapses_count['RELAPSES_COUNT'] > 1
        ]['DONOR_ID'].unique()
        X = historical_events_df[historical_events_df["DONOR_ID"].isin(subjects_with_multiple_relapses_ids)]
        X = X.sort_values(by=["DONOR_ID", "RELAPSE_START"])
    else:
        print("Using all sequences, including those with 1 step.")
        X = historical_events_df
        X = X.sort_values(by=["DONOR_ID", "RELAPSE_START"])

    return X, filepath_historical_events

def inference_and_plot_hmm_training_results(
    hmm: RecurrentHealthEventsHMM,
    X,
    save_dir,
    neptune_run=None,
    hidden_states_mapping=None,
    hidden_state_labels=None,
    hidden_state_labels_order=None,
    event_name="readmission",
    event_time_cat_col="READMISSION_TIME_CAT",
    time_cat_labels_order=None,
    continuous_event_time_col="LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION",
):
    """
    Apply HMM in inference mode and plots and saves HMM training results including transition matrix, inference results, and feature distributions.

    Parameters:
    - hmm: Trained HMM model.
    - X: Training data.
    - save_dir: Directory to save plots and results.
    - neptune_run: Neptune run object for logging (optional).
    - hidden_states_mapping: Mapping of hidden states to labels.
    - hidden_state_labels: List of hidden state labels.
    - hidden_state_labels_order: Order of hidden state labels for plotting.
    - event_name: Name of the event (e.g., "readmission").
    - event_time_cat_col: Column name for event time categories.
    - time_cat_labels_order: Order of event time categories for plotting.
    - continuous_event_time_col: Column name for continuous event time.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save transition matrix
    transition_matrix = hmm.get_transition_matrix()
    transition_matrix_df = pd.DataFrame(
        transition_matrix,
        index=[hidden_states_mapping[i] for i in range(len(transition_matrix))],
        columns=[hidden_states_mapping[i] for i in range(len(transition_matrix))],
    )
    transition_matrix_df = transition_matrix_df.loc[
        hidden_state_labels, hidden_state_labels
    ]
    title = f"Transition Matrix"
    filename = f"transition_matrix.png"
    fig = plot_transition_matrix_heatmap(
        transition_matrix_df,
        title=title,
        show_plot=False,
        save_file=os.path.join(save_dir, filename),
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, filename, fig, path="plots/training")

    # Inference
    pred_seq = hmm.infer_hidden_states(X)
    pred_proba = hmm.predict_proba(X)
    training_seq_with_probs_df = add_pred_state_to_df(
        X, pred_seq, labels_dict=hidden_states_mapping
    )
    training_seq_with_probs_df = add_pred_prob_states_to_df(
        X, pred_proba, labels=hidden_states_mapping
    )
    
    training_seq_with_probs_filepath = os.path.join(save_dir, "training_seq_with_probs.csv")
    training_seq_with_probs_df.to_csv(
        training_seq_with_probs_filepath, index=False
    )

    if neptune_run:
        upload_hmm_output_file_to_neptune(
            neptune_run,
            training_seq_with_probs_filepath
        )

    # Plots: Readmission time by hidden state, feature KDE by state
    filename = f"{event_name.lower()}_time_by_hidden_state.png"
    fig = plot_cat_event_time_by_hidden_state(
        training_seq_with_probs_df,
        title_suffix=" (Training HMM Sequences)",
        time_cat_col=event_time_cat_col,
        event_name=event_name,
        show_plot=False,
        hue_order=time_cat_labels_order,
        hidden_states_order=hidden_state_labels_order,
        save_file=os.path.join(save_dir, filename),
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, filename, fig, path="plots/training")

    filename = f"{continuous_event_time_col.lower()}_by_hidden_state.png"
    fig = plot_feature_kde_by_hidden_state(
        continuous_event_time_col,
        training_seq_with_probs_df,
        hidden_state_col="HEALTH_HIDDEN_RISK",
        hue_order=hidden_state_labels_order,
        show_plot=False,
        suffix_title="(Training HMM Sequences)",
        save_path=os.path.join(save_dir, filename),
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, filename, fig, path="plots/training")

def train_and_evaluate_hmm(
    hmm_config,
    X,
    initialize_from_first_obs_with_gmm=True,
    neptune_run=None,
    random_state=42,
):
    features = hmm_config["features"]
    model_name = hmm_config["model_name"]
    n_states = hmm_config["n_states"]
    hidden_state_labels = hmm_config["hidden_state_labels"]
    hidden_state_labels_order = sort_state_labels_by_severity(
        hidden_state_labels
    )  # Sort labels by severity if needed

    event_time_cat_col = hmm_config.get("event_time_cat_col", "READMISSION_TIME_CAT")
    time_cat_labels_order = hmm_config.get(
        "event_time_cat_order", ["0-30", "30-120", "120+"]
    )
    event_name = hmm_config.get("event_name", "readmission")
    continuous_event_time_col = hmm_config.get(
        "continuous_event_time_col", "LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION"
    )
    feature_define_state_labels = hmm_config.get("feature_define_state_labels", None)
    apply_power_transform = hmm_config.get("apply_power_transform", False)

    save_dir = os.path.join(hmm_config["save_model_path"], model_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Model name:", model_name)
    print("Event name:", event_name)
    print("Features used for training:", features)
    print("Number of states:", n_states)
    print("Hidden state labels:", hidden_state_labels)
    print("Feature to define Hidden State Labels: ", feature_define_state_labels)
    print("Event time category column: ", event_time_cat_col)
    print("Event time categories:", time_cat_labels_order)
    print("Continuous event time column: ", continuous_event_time_col)
    print("Saving model to:", save_dir)
    print("Using GMM for initialization:", initialize_from_first_obs_with_gmm)
    print("Using random state:", random_state)
    print("Training data shape:", X.shape)
    print("Applying power transform to features: ", apply_power_transform)        

    hmm = RecurrentHealthEventsHMM(hmm_config)

    if apply_power_transform:
        print("Applying power transform to features...")
        print(hmm.power_transform_columns)
        X = hmm.fit_transform_power_variables(X)

    # Train HMM
    hmm.train(
        X,
        verbose=True,
        random_state=random_state,
        initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm,
    )
    hmm.define_hidden_state_labels()
    hidden_states_mapping = hmm.get_hidden_state_labels()

    # Save model & parameters
    hmm.save_model_params()
    model_file_path = hmm.save_model()
    hmm.save_model_metrics(X)

    if neptune_run:
        final_model_params = hmm.get_model_params_str()
        neptune_run["parameters"] = final_model_params
        upload_model_to_neptune(
            neptune_run,
            model_file_path
        )

    # Save feature distributions
    distributions_params_df = hmm.get_features_dist_df(True).sort_values(
        by="State Label",
        key=lambda col: [hidden_state_labels.index(label) for label in col],
    )
    for feature, dist in hmm.features.items():
        dist_type = DistributionType(dist)
        if dist_type not in [DistributionType.CATEGORICAL, DistributionType.BERNOULLI]:
            filename = f"pred_{feature.lower()}_dist.png"
            save_path = os.path.join(save_dir, filename)
            fig = plot_feature_distribution_per_hidden_state(
                feature,
                distributions_params_df,
                hue_order=hidden_state_labels_order,
                save_file=save_path,
                show_plot=False,
            )
            if neptune_run:
                add_plot_to_neptune_run(
                    neptune_run, filename, fig, path="plots/training"
                )

    # Save parameters DataFrame
    create_and_save_params_df(
        hmm,
        labels=hidden_state_labels,
        save_path=os.path.join(save_dir, "hmm_params.html"),
        include_state_labels=True,
        neptune_run=neptune_run,
        neptune_path="hmm_params",
    )

    inference_and_plot_hmm_training_results(
        hmm,
        X,
        save_dir,
        neptune_run=neptune_run,
        hidden_states_mapping=hidden_states_mapping,
        hidden_state_labels=hidden_state_labels,
        hidden_state_labels_order=hidden_state_labels_order,
        event_name=event_name,
        event_time_cat_col=event_time_cat_col,
        time_cat_labels_order=time_cat_labels_order,
        continuous_event_time_col=continuous_event_time_col,
    )

    # Main metrics
    log_likelihood, num_sequences = hmm.log_likelihood(X)
    metrics = {
        "aic": hmm.calculate_aic(X),
        "bic": hmm.calculate_bic(X),
        "log_likelihood": log_likelihood,
        "num_sequences": num_sequences,
        "total_num_params": hmm.get_total_num_params(),
    }
    if neptune_run:
        for metric_name, value in metrics.items():
            neptune_run[f"metrics/{metric_name}"] = value

    return {
        "metrics": metrics,
        "model_save_dir": save_dir,
    }, hmm

def run_model_selection(X: pd.DataFrame, hmm_config_path: str, save_dir: str, max_num_states=11, initialize_from_first_obs_with_gmm=True):
    """
    Trains HMMs for different numbers of states and saves AIC/BIC plots.
    """
    with open(hmm_config_path, 'r') as file:
        hmm_config = yaml.safe_load(file)
    print("Configuration loaded from YAML file: ", hmm_config_path)

    model_name = hmm_config["model_name"]
    save_dir = os.path.join(hmm_config["save_model_path"], model_name)
    os.makedirs(save_dir, exist_ok=True)

    print("Model name: ", model_name)
    print("Model selection will be performed for up to ", max_num_states, " states.")

    results = get_model_selection_results_hmm(
        max_num_states, hmm_config, X, initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm
    )
    num_params_list = results["num_params"]
    aic_values = results["aic"]
    bic_values = results["bic"]
    num_states_list = list(range(2, max_num_states + 1))

    # Save AIC plot
    fig_aic = go.Figure()
    fig_aic.add_trace(go.Scatter(x=num_params_list, y=aic_values, mode='lines+markers', name='AIC'))
    fig_aic.update_layout(
        title="AIC vs Number of Parameters",
        xaxis_title="Number of Parameters",
        yaxis_title="AIC",
        template="plotly_white"
    )
    aic_path = os.path.join(save_dir, "aic.html")
    fig_aic.write_html(aic_path)

    # Save BIC plot
    fig_bic = go.Figure()
    fig_bic.add_trace(go.Scatter(x=num_params_list, y=bic_values, mode='lines+markers', name='BIC'))
    fig_bic.update_layout(
        title="BIC vs Number of Parameters",
        xaxis_title="Number of Parameters",
        yaxis_title="BIC",
        template="plotly_white"
    )
    bic_path = os.path.join(save_dir, "bic.html")
    fig_bic.write_html(bic_path)

    # Return summary (optional)
    summary = {
        "aic_values": aic_values,
        "bic_values": bic_values,
        "num_params_list": num_params_list,
        "num_states_list": num_states_list,
        "aic_min": (num_states_list[int(np.argmin(aic_values))], aic_values[int(np.argmin(aic_values))]),
        "bic_min": (num_states_list[int(np.argmin(bic_values))], bic_values[int(np.argmin(bic_values))]),
        "aic_path": aic_path,
        "bic_path": bic_path,
    }
    return summary

def get_training_sequences_stats(
    X: pd.DataFrame, subject_id_col: str = "SUBJECT_ID", event_id_col: str = "HADM_ID"
):
    sequence_stats_per_subject_df = X.groupby(subject_id_col).agg(
        seq_length=(event_id_col, "nunique")
    )

    min_length = sequence_stats_per_subject_df["seq_length"].min()
    max_length = sequence_stats_per_subject_df["seq_length"].max()
    mean_length = sequence_stats_per_subject_df["seq_length"].mean()
    median_length = sequence_stats_per_subject_df["seq_length"].median()
    num_sequences_length_2 = (sequence_stats_per_subject_df["seq_length"] == 2).sum()
    num_sequences_length_1 = (sequence_stats_per_subject_df["seq_length"] == 1).sum()

    results = {
        "min_length": min_length,
        "max_length": max_length,
        "mean_length": mean_length,
        "median_length": median_length,
        "num_sequences_length_2": num_sequences_length_2,
        "num_sequences_length_1": num_sequences_length_1,
        "num_sequences": len(sequence_stats_per_subject_df),
    }

    return results

def sort_state_labels_by_severity(labels: list):
    # Define severity for each category
    severity_order = {
        'high': 3,
        'medium': 2,
        'low': 1
    }
    def label_key(label):
        # Example label: 'high_2'
        # Support labels without _index (e.g. just 'high')
        parts = label.split('_')
        category = parts[0]
        index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (severity_order.get(category, 0), index)
    # Sort by: category (descending), index (descending)
    return sorted(labels, key=label_key, reverse=True)

def summarize_hidden_state_counts_from_df(
    df: pd.DataFrame,
    hmm_model: RecurrentHealthEventsHMM,
    subject_id_col: str = "SUBJECT_ID",
    hidden_state_col_name="HEALTH_HIDDEN_RISK",
    final_col_middle_str="HIDDEN_RISK",
):
    """
    Adds inferred hidden states to df, then summarizes the number of times spent in each hidden state per subject.
    Returns a DataFrame with one row per subject and columns for each hidden state.
    """
    hidden_state_labels_dict = hmm_model.get_hidden_state_labels()

    # 1. Infer hidden state sequence for all rows
    pred_seq = hmm_model.infer_hidden_states(df)
    df = add_pred_state_to_df(
        df, pred_seq, col_name=hidden_state_col_name, labels_dict=hidden_state_labels_dict
    )

    # 2. Get state labels
    unique_states = hidden_state_labels_dict.values()

    # 3. Group by subject and count occurrences of each state
    state_counts = (
        df.groupby(subject_id_col)[hidden_state_col_name]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    # 4. Optionally rename columns for clarity
    state_counts = state_counts.rename(
        columns={
            s: f"PAST_COUNT_{final_col_middle_str.upper()}_{s.upper().replace(' ', '_')}"
            for s in unique_states
        }
    )
    for col in state_counts.columns:
        if col.startswith(f"PAST_COUNT_{final_col_middle_str.upper()}_"):
            state_counts[col] = state_counts[col].astype(int)

    return state_counts
