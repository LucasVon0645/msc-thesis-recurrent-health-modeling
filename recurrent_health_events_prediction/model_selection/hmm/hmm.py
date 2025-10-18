import json
from typing import Optional
import neptune
import pandas as pd
import yaml
import os
import plotly.graph_objects as go
from recurrent_health_events_prediction.utils.general_utils import check_if_directory_exists, check_if_file_exists, import_yaml_config

from recurrent_health_events_prediction.model.RecurrentHealthEventsHMM import get_model_selection_results_hmm
from recurrent_health_events_prediction.model_selection.hmm_utils import find_best_num_states, log_hmm_selection_config_to_neptune, plot_model_selection_results_hmm, print_model_selection_config
from recurrent_health_events_prediction.training.utils_hmm import get_training_sequences_stats, load_and_prepare_historical_data_mimic, load_and_prepare_historical_data_relapse
from recurrent_health_events_prediction.utils.neptune_utils import add_plotly_plots_to_neptune_run, initialize_neptune_run, track_file_in_neptune
import time


def model_selection_hmm(
    hmm_config_path: str,
    data_config_path: str,
    dataset: str,
    training_data_path: str,
    max_num_states: int,
    n_repeats=10,
    max_attempts_per_fit=7,
    log_in_neptune=False,
    show_plot: Optional[bool] = False,
    neptune_tags=["hmm_model_selction"],
):
    hmm_config = import_yaml_config(hmm_config_path)
    model_name = hmm_config.get("model_name", "hmm_model")
    base_dir = hmm_config.get("save_model_path", "_models/" + dataset + "/hmm")
    save_dir = os.path.join(base_dir, model_name)

    use_only_sequences_gte_2_steps = hmm_config.get("use_only_sequences_gte_2_steps", False)
    initialize_from_first_obs_with_gmm = not use_only_sequences_gte_2_steps

    print_model_selection_config(hmm_config, max_num_states, n_repeats, max_attempts_per_fit, use_only_sequences_gte_2_steps)

    run_name = f"{model_name.lower()}_selection"
    neptune_run = initialize_neptune_run(data_config_path, run_name=run_name, dataset=dataset, tags=neptune_tags) if log_in_neptune else None

    # Load data
    if dataset == "mimic":
        X, filepath_historical_events = load_and_prepare_historical_data_mimic(training_data_path, use_only_sequences_gt_2_steps=use_only_sequences_gte_2_steps)
    elif dataset == "synthetic":
        filepath_historical_events = os.path.join(training_data_path, "synthetic_test.csv")
        X = pd.read_csv(filepath_historical_events)
    elif dataset == "relapse":
        X, filepath_historical_events = load_and_prepare_historical_data_relapse(training_data_path, use_only_sequences_gte_2_steps)

    if log_in_neptune:
        log_hmm_selection_config_to_neptune(
            neptune_run,
            hmm_config,
            max_num_states,
            n_repeats,
            max_attempts_per_fit,
            use_only_sequences_gte_2_steps,
        )
        # Log model config and training data path
        neptune_run["training_data/path"] = training_data_path
        # Log training data statistics
        training_data_stats = get_training_sequences_stats(
            X,
            event_id_col=hmm_config["event_id_col"],
            subject_id_col=hmm_config["id_col"],
        )
        neptune_run["training_data/stats"] = training_data_stats
        neptune_tracking_path = "training_data/files/historical_relapses"
        track_file_in_neptune(
            neptune_run, neptune_tracking_path, filepath_historical_events
        )

    start_time = time.time()
    hmm_model_selection_results = get_model_selection_results_hmm(
        max_num_states,
        hmm_config,
        X,
        initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm,
        n_repeats=n_repeats,
        max_attempts_per_fit=max_attempts_per_fit,
    )
    # Measure elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    elapsed_time_str = f"{elapsed_minutes}min{elapsed_seconds:02d}s"
    print(f"Model selection completed in {elapsed_time_str}.")

    # Main results
    num_params_list = hmm_model_selection_results["num_params"]
    num_states_list = hmm_model_selection_results["num_states"]
    bic_mean = hmm_model_selection_results["bic_mean"]
    aic_mean = hmm_model_selection_results["aic_mean"]
    bic_std = hmm_model_selection_results["bic_std"]
    aic_std = hmm_model_selection_results["aic_std"]

    aic_results_dict = find_best_num_states("aic", aic_mean, aic_std, num_states_list)
    bic_results_dict = find_best_num_states("bic", bic_mean, bic_std, num_states_list)

    if log_in_neptune:
        neptune_run["model_selection/aic"] = aic_results_dict
        neptune_run["model_selection/bic"] = bic_results_dict
        neptune_run["model_selection/all_results"] = json.dumps(hmm_model_selection_results)
        neptune_run["model_selection/elapsed_time"] = elapsed_time_str

    # Plot BIC and AIC
    plot_bic = plot_model_selection_results_hmm("BIC", num_params_list, num_states_list, bic_mean, bic_std, save_dir=save_dir, show_plot=show_plot)
    plot_aic = plot_model_selection_results_hmm("AIC", num_params_list, num_states_list, aic_mean, aic_std, save_dir=save_dir, show_plot=show_plot)

    if log_in_neptune:
        add_plotly_plots_to_neptune_run(neptune_run, plot_bic, "bic.html", filepath="plots")
        add_plotly_plots_to_neptune_run(neptune_run, plot_aic, "aic.html", filepath="plots")
        neptune_run.stop()

    return aic_results_dict, bic_results_dict, hmm_model_selection_results


def main():
    dataset = "mimic"  # "mimic", "synthetic", or "relapse"
    max_num_states = 4 # Maximum number of states to consider for model selection
    n_repeats = 5 # Number of repeats for each state, used to average the results
    max_attempts_per_fit = 6 # Number of attempts to fit the model per state
    log_in_neptune = True # Set to True to log in Neptune
    data_config_path = "/workspaces/master-thesis-recurrent-health-events-prediction/recurrent_health_events_prediction/configs/data_config.yaml"
    neptune_tags = ["hmm_model_selection", "multiple_hosp_patients"]

    if dataset == "mimic":
        model_name = "hmm_mimic_time_log_gamma"
        hmm_config_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/mimic/hmm/{model_name}/{model_name}_config.yaml"
        training_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/"
    elif dataset == "relapse":
        model_name = "hmm_binary_relapse_30_days"
        hmm_config_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/drug_relapse/hmm/{model_name}/{model_name}_config.yaml"
        training_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/"
    
    model_config_exists = check_if_file_exists(hmm_config_path)
    training_data_exists = check_if_directory_exists(training_data_path)

    if not model_config_exists:
        raise FileNotFoundError(f"Model configuration file not found: {hmm_config_path}")
    if not training_data_exists:
        raise FileNotFoundError(f"Training data directory not found: {training_data_path}")
    
    model_selection_hmm(
        hmm_config_path=hmm_config_path,
        data_config_path=data_config_path,
        dataset=dataset,
        training_data_path=training_data_path,
        max_num_states=max_num_states,
        n_repeats=n_repeats,
        max_attempts_per_fit=max_attempts_per_fit,
        log_in_neptune=log_in_neptune,
        neptune_tags=neptune_tags,
    )

if __name__ == "__main__":
    main()
