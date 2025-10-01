import os
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import (
    RecurrentHealthEventsHMM,
)
from recurrent_health_events_prediction.model.utils import load_model
from recurrent_health_events_prediction.preprocessing.gen_dataset_hmm_utils import (
    filter_last_events_mimic,
    apply_set_observation_window_mimic,
    hide_time_to_next_event_hmm_feature,
    get_events_up_to_event,
    plot_event_duration_kde_by_hidden_state,
    plot_facet_hidden_state_distribution,
    save_df,
    compute_hidden_state_probs,
    build_hidden_state_results,
    run_chi2_and_plot,
)
from recurrent_health_events_prediction.preprocessing.utils import (
    get_past_events,
    hot_encode_drug_classes,
    remap_discharge_location,
)
from recurrent_health_events_prediction.training.utils import preprocess_features_to_one_hot_encode
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_plot_to_neptune_run,
    upload_hmm_output_file_to_neptune,
)
from recurrent_health_events_prediction.training.utils_hmm import (
    plot_cat_event_time_by_hidden_state,
    plot_feature_kde_by_hidden_state,
    sort_state_labels_by_severity,
    summarize_hidden_state_counts_from_df,
)

# ========================
# USER CONFIGURABLE BLOCK
# ========================
MODEL_NAME = "hmm_binary_relapse_90_days"
BASE_MODEL_PATH = "/workspaces/master-thesis-recurrent-health-events-prediction/_models/drug_relapse/hmm"
TRAINING_DATA_PATH = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/relapse_cleaned"
SKIP_FILTERING = True
DATASET = "relapse"


def load_hmm_model():
    model_pickle_path = os.path.join(BASE_MODEL_PATH, MODEL_NAME, f"{MODEL_NAME}.pkl")
    hmm_model: RecurrentHealthEventsHMM = load_model(model_pickle_path)
    hmm_config = hmm_model.config

    print(f"Loaded HMM model from: {model_pickle_path}")
    print(f"HMM configuration: {hmm_config}")

    return hmm_model


def load_data_for_inference_mimic(inference_data_path):
    """
    Loads data for inference from the specified path in the data configuration.

    Args:
        inference_data_path (str): Path to the inference data directory.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    # Load all events
    filepath_all_events = os.path.join(inference_data_path, "all_events.csv")
    all_events_df = pd.read_csv(filepath_all_events)
    all_events_df["ADMITTIME"] = pd.to_datetime(all_events_df["ADMITTIME"])
    all_events_df["DISCHTIME"] = pd.to_datetime(all_events_df["DISCHTIME"])
    all_events_df = remap_discharge_location(all_events_df)
    all_events_df, new_cols = preprocess_features_to_one_hot_encode(
        all_events_df,
        ["DISCHARGE_LOCATION"],
        one_hot_cols_to_drop=["DISCHARGE_LOCATION_OTHERS"],
    )
    all_events_df[new_cols] = all_events_df[new_cols].astype(int)

    # Load last events
    filepath_last_events = os.path.join(inference_data_path, "last_events.csv")
    last_events_df = pd.read_csv(filepath_last_events)
    last_events_df["ADMITTIME"] = pd.to_datetime(last_events_df["ADMITTIME"])
    last_events_df["DISCHTIME"] = pd.to_datetime(last_events_df["DISCHTIME"])
    last_events_df = remap_discharge_location(last_events_df)
    last_events_df, new_cols = preprocess_features_to_one_hot_encode(
        last_events_df,
        ["DISCHARGE_LOCATION"],
        one_hot_cols_to_drop=["DISCHARGE_LOCATION_OTHERS"],
    )
    last_events_df[new_cols] = last_events_df[new_cols].astype(int)


    print("Load all_events and last_events dataframes")
    print(f"All events CSV: {inference_data_path + '/all_events.csv'}")
    print(f"Last events CSV: {inference_data_path + '/last_events.csv'}")

    return all_events_df, last_events_df, filepath_all_events, filepath_last_events


def load_data_for_inference_relapse(inference_data_path):
    """
    Loads data for inference from the specified path in the data configuration.

    Args:
        inference_data_path (str): Path to the inference data directory.

    Returns:
        tuple: Two DataFrames containing all events and last events data, in this order.
    """
    # Load all events
    filepath_all_events = os.path.join(inference_data_path, "all_relapses.csv")
    all_events_df = pd.read_csv(filepath_all_events)
    all_events_df["RELAPSE_START"] = pd.to_datetime(all_events_df["RELAPSE_START"])
    all_events_df = hot_encode_drug_classes(all_events_df, "PREV_POSITIVE_DRUGS")

    # Load last events
    filepath_last_events = os.path.join(inference_data_path, "last_relapses.csv")
    last_events_df = pd.read_csv(filepath_last_events)
    last_events_df["RELAPSE_START"] = pd.to_datetime(last_events_df["RELAPSE_START"])
    last_events_df = hot_encode_drug_classes(last_events_df, "PREV_POSITIVE_DRUGS")

    print("Load all_events and last_events dataframes")
    print(f"All events CSV: {filepath_all_events}")
    print(f"Last events CSV: {filepath_last_events}")

    return all_events_df, last_events_df, filepath_all_events, filepath_last_events


def process_mimic_events(all_events_df, last_events_df, skip_filtering):
    if not skip_filtering:
        last_events_df = filter_last_events_mimic(last_events_df)
        all_events_df = apply_set_observation_window_mimic(all_events_df)
    return all_events_df, last_events_df


def extract_partial_trajectories(
    all_events_df,
    last_events_df,
    time_feature_col=None,
    event_name="readmission",
    event_id_col="HADM_ID",
    subject_id_col="SUBJECT_ID",
    time_col="ADMITTIME",
):
    event_col = f"{event_name.upper()}_EVENT"
    last_observed_event_ids = last_events_df[last_events_df[event_col] == 1][
        [subject_id_col, event_id_col]
    ].set_index(subject_id_col)[event_id_col]
    last_censored_event_ids = last_events_df[last_events_df[event_col] == 0][
        [subject_id_col, event_id_col]
    ].set_index(subject_id_col)[event_id_col]

    events_up_to_last_observed_df = get_events_up_to_event(
        all_events_df,
        last_observed_event_ids,
        event_id_col=event_id_col,
        time_col=time_col,
        id_col=subject_id_col,
    )
    events_up_to_last_censored_df = get_events_up_to_event(
        all_events_df,
        last_censored_event_ids,
        event_id_col=event_id_col,
        time_col=time_col,
        id_col=subject_id_col,
    )

    if time_feature_col is not None:
        events_up_to_last_observed_df = hide_time_to_next_event_hmm_feature(
            events_up_to_last_observed_df,
            event_id_col,
            last_observed_event_ids,
            time_feature_col,
        )
        events_up_to_last_censored_df = hide_time_to_next_event_hmm_feature(
            events_up_to_last_censored_df,
            event_id_col,
            last_censored_event_ids,
            time_feature_col,
        )

    return (
        last_observed_event_ids,
        last_censored_event_ids,
        events_up_to_last_observed_df,
        events_up_to_last_censored_df,
    )


def infer_last_hidden_states(
    hmm_model: RecurrentHealthEventsHMM,
    last_events_df,
    last_readmission_event_ids,
    last_censored_event_ids,
    events_up_to_last_readmission_df,
    events_up_to_last_censored_df,
) -> pd.DataFrame:
    """
    Infers hidden states for the last events and merges them with the last events DataFrame.
    Parameters:
    - hmm_model: The trained HMM model.
    - last_events_df: DataFrame containing the last events for each subject.
    - last_readmission_event_ids: Series of last readmission event IDs.
    - last_censored_event_ids: Series of last censored event IDs.
    - events_up_to_last_readmission_df: DataFrame of events up to the last readmission event with masked time feature.
    - events_up_to_last_censored_df: DataFrame of events up to the last censored event with masked time feature.
    Returns:
    - merged_df: DataFrame containing last events with inferred hidden states.
    """
    event_id_col = hmm_model.config.get("event_id_col", "HADM_ID")

    # Compute probabilities and map states
    probs_readm = compute_hidden_state_probs(
        hmm_model, events_up_to_last_readmission_df
    )
    probs_cens = compute_hidden_state_probs(hmm_model, events_up_to_last_censored_df)

    idx_to_labels = hmm_model.get_hidden_state_labels()
    labels_to_idx = {label: index for index, label in idx_to_labels.items()}
    results_readm = build_hidden_state_results(
        last_readmission_event_ids.values,
        probs_readm,
        idx_to_labels,
        labels_to_idx,
        event_id_col=event_id_col,
    )
    results_cens = build_hidden_state_results(
        last_censored_event_ids.values,
        probs_cens,
        idx_to_labels,
        labels_to_idx,
        event_id_col=event_id_col,
    )
    last_hidden_states_df = pd.concat([results_readm, results_cens], ignore_index=True)

    merged_df = last_events_df.merge(last_hidden_states_df, on=event_id_col, how="left")

    return merged_df


def infer_past_states_add_stats(
    hmm_model: RecurrentHealthEventsHMM,
    last_events_with_hmm_features_df,
    events_up_to_last_obs_df,
    events_up_to_last_censored_df,
    subject_id_col,
):
    past_events_obs_df = get_past_events(events_up_to_last_obs_df, subject_id_col)
    past_events_cens_df = get_past_events(events_up_to_last_censored_df, subject_id_col)

    past_hidden_states_obs_count_df = summarize_hidden_state_counts_from_df(
        past_events_obs_df, hmm_model, subject_id_col=subject_id_col
    )
    past_hidden_states_cens_count_df = summarize_hidden_state_counts_from_df(
        past_events_cens_df, hmm_model, subject_id_col=subject_id_col
    )

    past_hidden_states_count_df = pd.concat(
        [past_hidden_states_obs_count_df, past_hidden_states_cens_count_df],
        ignore_index=True,
    )

    last_events_with_hmm_features_df = last_events_with_hmm_features_df.merge(
        past_hidden_states_count_df, on=subject_id_col, how="left"
    )
    past_count_cols = [
        col
        for col in last_events_with_hmm_features_df.columns
        if col.startswith("PAST_COUNT")
    ]
    last_events_with_hmm_features_df[past_count_cols] = (
        last_events_with_hmm_features_df[past_count_cols].fillna(0).astype(int)
    )

    return last_events_with_hmm_features_df


def save_outputs_and_plots(
    save_dir,
    last_events_with_hmm_feat_df,
    events_up_to_last_readmission_df,
    events_up_to_last_censored_df,
    time_cat_labels_order,
    hidden_state_label_order,
    event_name="readmission",
    event_time_cat_col="READMISSION_TIME_CAT",
    continuous_event_time_col="LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION",
    neptune_run=None,
):

    last_events_with_hmm_feat_df = last_events_with_hmm_feat_df.reset_index()
    events_up_to_last_readmission_df = events_up_to_last_readmission_df.reset_index()
    events_up_to_last_censored_df = events_up_to_last_censored_df.reset_index()

    # Save partial trajectories
    fname = "events_up_to_last_readmission.csv"
    save_df(events_up_to_last_readmission_df, os.path.join(save_dir, fname))
    fname = "events_up_to_last_censored.csv"
    save_df(events_up_to_last_censored_df, os.path.join(save_dir, fname))
    # Save events with hidden states
    output_hmm_data_filepath = os.path.join(
        save_dir, "last_events_with_hidden_states.csv"
    )
    save_df(last_events_with_hmm_feat_df, output_hmm_data_filepath)
    if neptune_run:
        upload_hmm_output_file_to_neptune(
            neptune_run,
            output_hmm_data_filepath,
            base_neptune_path="artifacts/hmm_output",
        )
    # Run chi2/plots
    results_str, fig = run_chi2_and_plot(
        last_events_with_hmm_feat_df, save_dir, event_time_cat_col
    )
    if neptune_run:
        add_plot_to_neptune_run(
            neptune_run,
            "target_health_hidden_risk_vs_event_time_cat.png",
            fig,
            path="plots/inference",
        )
        neptune_run["inference/chi2_independence_results"] = results_str

    # -- 1. Plot: Target Event Time Cat by Hidden State
    filename = f"target_{event_name}_time_cat_by_hidden_state.png"
    filepath = os.path.join(save_dir, filename)
    print(f"Saving plot to {filepath}")
    fig = plot_cat_event_time_by_hidden_state(
        last_events_with_hmm_feat_df,
        time_cat_col=event_time_cat_col,
        title_suffix=" (Target Event)",
        event_name=event_name,
        hue_order=time_cat_labels_order,
        save_file=filepath,
        hidden_states_order=hidden_state_label_order,
        show_plot=False,
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, filename, fig, path="plots/inference")

    # -- 2. Plot: KDE of Continuous Event Time Feature by Hidden State
    filename = f"{continuous_event_time_col.lower()}_by_hidden_state_last_events.png"
    fig = plot_feature_kde_by_hidden_state(
        continuous_event_time_col,
        last_events_with_hmm_feat_df,
        hidden_state_col="HEALTH_HIDDEN_RISK",
        hue_order=hidden_state_label_order,
        show_plot=False,
        suffix_title=f"(Inference HMM Last {event_name.title()} Events)",
        save_path=os.path.join(save_dir, filename),
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, filename, fig, path="plots/inference")

    # -- 3. Plot: KDE of Event Duration by Hidden State
    duration_kde_path = os.path.join(
        save_dir, "target_kde_event_duration_by_hidden_state.png"
    )
    fig = plot_event_duration_kde_by_hidden_state(
        df=last_events_with_hmm_feat_df,
        event_name=event_name,
        hidden_state_label_order=hidden_state_label_order,
        save_path=duration_kde_path,
    )
    print(f"KDE plot saved to {duration_kde_path}")
    if neptune_run:
        add_plot_to_neptune_run(
            neptune_run,
            "target_kde_event_duration_by_hidden_state.png",
            fig,
            path="plots/inference",
        )

    # -- 4. Plot: Histogram of Health Hidden Risk by Event
    hist_path = os.path.join(
        save_dir, f"target_hist_hidden_state_by_{event_name}_event.png"
    )
    upper_limit = np.ceil(last_events_with_hmm_feat_df["EVENT_DURATION"].max()).item()
    fig = plot_facet_hidden_state_distribution(
        last_events_with_hmm_feat_df,
        save_path=hist_path,
        hue_order=hidden_state_label_order,
        axis_order=hidden_state_label_order,
        event_name=event_name,
        title=f"Histogram of Health Hidden Risk by {event_name.title()} Event ({int(upper_limit)} days)",
    )
    if neptune_run:
        add_plot_to_neptune_run(
            neptune_run,
            "target_hist_hidden_state_by_event.png",
            fig,
            path="plots/inference",
        )

    print(f"All outputs saved to: {save_dir}")


def generate_dataset_hmm_feat(
    all_events_df, last_events_df, hmm_model: RecurrentHealthEventsHMM, neptune_run=None
):
    """
    Generates a survival dataset with HMM features from the provided events data.
    It takes all events in `all_events_df` up to events listed in `last_events_df`

    Args:
        all_events_df (pd.DataFrame): DataFrame containing all events data.
        last_events_df (pd.DataFrame): DataFrame containing the last events data.
        hmm_model: The trained HMM model.
        neptune_run: Neptune run object for logging (optional).

    Returns:
        pd.DataFrame: DataFrame containing the last events with HMM features.
    """
    hmm_config = hmm_model.config
    model_name = hmm_config["model_name"]
    event_name = hmm_config.get("event_name", "readmission")
    event_time_feature_col = hmm_config.get("event_time_feature_col", None)
    event_time_cat_col = hmm_config.get("event_time_cat_col", "READMISSION_TIME_CAT")
    continuous_event_time_col = hmm_config.get(
        "continuous_event_time_col", "LOG_DAYS_UNTIL_NEXT_HOSPITALIZATION"
    )
    event_time_cat_order = hmm_config.get(
        "event_time_cat_order", ["0-30", "30-120", "120+"]
    )
    event_id_col = hmm_config.get("event_id_col", "HADM_ID")
    subject_id_col = hmm_config.get("id_col", "SUBJECT_ID")
    time_col = hmm_config.get("time_col", "ADMITTIME")
    hidden_states_labels_order = sort_state_labels_by_severity(
        hmm_config.get("hidden_state_labels", None)
    )
    apply_power_transform = hmm_config.get("apply_power_transform", False)
    save_dir = os.path.join(hmm_config["save_model_path"], model_name)

    # Step 2: extract partial trajectories and mask event_time_feature_col
    (
        last_readmission_event_ids,
        last_censored_event_ids,
        events_up_to_last_readmission_df,
        events_up_to_last_censored_df,
    ) = extract_partial_trajectories(
        all_events_df,
        last_events_df,
        event_time_feature_col,
        event_name=event_name,
        event_id_col=event_id_col,
        subject_id_col=subject_id_col,
        time_col=time_col,
    )

    if apply_power_transform:
        print("Applying fitted power transform to features of historical sequences...")
        events_up_to_last_readmission_df = hmm_model.transform_with_fitted_power(
            events_up_to_last_readmission_df
        )
        events_up_to_last_censored_df = hmm_model.transform_with_fitted_power(
            events_up_to_last_censored_df
        )

    # Step 3: infer hidden states and merge
    last_events_with_hmm_feat_df = infer_last_hidden_states(
        hmm_model,
        last_events_df,
        last_readmission_event_ids,
        last_censored_event_ids,
        events_up_to_last_readmission_df,
        events_up_to_last_censored_df,
    )

    # Step 4: infer past states and add stats
    last_events_with_hmm_feat_df = infer_past_states_add_stats(
        hmm_model,
        last_events_with_hmm_feat_df,
        events_up_to_last_readmission_df,
        events_up_to_last_censored_df,
        subject_id_col=subject_id_col,
    )

    # Step 4: save outputs and plots
    save_outputs_and_plots(
        save_dir,
        last_events_with_hmm_feat_df,
        events_up_to_last_readmission_df,
        events_up_to_last_censored_df,
        event_time_cat_order,
        hidden_states_labels_order,
        event_name=event_name,
        event_time_cat_col=event_time_cat_col,
        continuous_event_time_col=continuous_event_time_col,
        neptune_run=neptune_run,
    )
    print(f"All outputs and plots are saved to: {save_dir}")

    return last_events_with_hmm_feat_df


def main():
    hmm_model = load_hmm_model()
    if DATASET == "mimic":
        all_events_df, last_events_df, filepath_all_events, filepath_last_events = (
            load_data_for_inference_mimic(TRAINING_DATA_PATH)
        )
        all_events_df, last_events_df = process_mimic_events(
            all_events_df, last_events_df, SKIP_FILTERING
        )
    else:
        all_events_df, last_events_df, filepath_all_events, filepath_last_events = (
            load_data_for_inference_relapse(TRAINING_DATA_PATH)
        )

    df = generate_dataset_hmm_feat(all_events_df, last_events_df, hmm_model)

    return df


if __name__ == "__main__":
    main()
