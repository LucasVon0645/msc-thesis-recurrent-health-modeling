import os
from typing import Optional
import numpy as np
import plotly.graph_objects as go

from recurrent_health_events_prediction.utils.general_utils import stringify_dict_values

def plot_model_selection_results_hmm(metric: str, num_params_list: list[int], num_states_list: list[int],
                                     mean_vales: list[float], std_values: list[float],
                                     save_dir: Optional[str] = None, show_plot=True):
    """
    Plots the model selection results for HMM based on the specified metric.
    Parameters:
    - metric: The metric to plot (e.g., 'bic', 'aic').
    - num_params_list: List of number of parameters for each model.
    - num_states_list: List of number of states for each model.
    - mean_vales: List of mean values for the metric.
    - std_values: List of standard deviation values for the metric.
    - save_dir: Directory to save the plot. If None, the plot will not be saved.
    - show_plot: Whether to display the plot.
    Returns:
    - fig: The plotly figure object.
    """
    metric = metric.upper()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=num_params_list,
        y=mean_vales,
        mode='lines+markers',
        hovertemplate=(
            metric + ": %{y:.2f}<br>"
            "Number of Parameters: %{x}<br>"
            "Number of States: %{customdata}"
        ),
        name=metric,
        error_y=dict(
            type='data',
            array=std_values,
            visible=True,
            thickness=2,  # optional, makes lines a bit thicker
            width=6       # optional, makes cap wider
        ),
        customdata=num_states_list  # Pass number of states as custom data
    ))

    fig.update_layout(
        title=f"{metric} vs Number of Parameters",
        xaxis_title="Number of Parameters",
        yaxis_title=metric,
        template="plotly_white"
    )
    if save_dir:
        filepath = os.path.join(save_dir, f"{metric.lower()}.html")
        fig.write_html(filepath)
    if show_plot:
        fig.show()
    return fig

def find_best_num_states(metric_name: str, mean_values: list, mean_std: list, num_states_list: list):
    # Filter out NaN or None values
    valid_indices = [i for i, value in enumerate(mean_values) if not (np.isnan(value) or value is None)]
    
    if not valid_indices:
        raise ValueError(f"All values for {metric_name} are NaN or None.")
    
    # Find the index of the minimum valid value
    valid_mean_values = [mean_values[i] for i in valid_indices]
    arg_min_metric = valid_indices[np.argmin(valid_mean_values)]
    
    print(f"Minimum {metric_name}: {mean_values[arg_min_metric]} at {num_states_list[arg_min_metric]} states")

    results = {
        "metric_name": metric_name,
        "best_num_states": num_states_list[arg_min_metric],
        "best_metric_value": mean_values[arg_min_metric],
        "best_metric_std": mean_std[arg_min_metric]
    }

    return results

def log_hmm_selection_config_to_neptune(neptune_run, hmm_config, max_num_states, n_repeats,
                              max_attempts_per_fit, use_only_sequences_gte_2_steps):
    """
    Logs HMM selection configuration details to a Neptune run.

    Parameters:
    - neptune_run: Neptune run object to log the configuration.
    - hmm_config: Dictionary containing HMM configuration details.
    - max_num_states: Maximum number of states for the HMM model.
    - n_repeats: Number of repeats for model selection.
    - max_attempts_per_fit: Maximum attempts per fit during training.
    - use_only_sequences_gte_2_steps: Boolean indicating whether to use sequences with >= 2 steps.
    """
    print("Logging HMM selection configuration to Neptune...")
    neptune_config_log = {
        "model_name": hmm_config["model_name"],
        "max_num_states": max_num_states,
        "n_repeats": n_repeats,
        "max_attempts_per_fit": max_attempts_per_fit,
        "use_only_sequences_gte_2_steps": use_only_sequences_gte_2_steps,
        "features": hmm_config["features"]
    }
    neptune_run["config"] = neptune_config_log
    neptune_run["emission_variables"] = stringify_dict_values(hmm_config["features"])

def print_model_selection_config(hmm_config, max_num_states, n_repeats,
                              max_attempts_per_fit, use_only_sequences_gte_2_steps):
    """
    Prints the HMM selection configuration details to the console.
    Parameters:
    - hmm_config: Dictionary containing HMM configuration details.
    - max_num_states: Maximum number of states for the HMM model.
    - n_repeats: Number of repeats for model selection.
    - max_attempts_per_fit: Maximum attempts per fit during training.
    - use_only_sequences_gte_2_steps: Boolean indicating whether to use sequences with >= 2 steps.
    """
    print("HMM Selection Configuration:")
    print(f"Model Name: {hmm_config['model_name']}")
    print(f"Max Number of States: {max_num_states}")
    print(f"Number of Repeats: {n_repeats}")
    print(f"Max Attempts per Fit: {max_attempts_per_fit}")
    print(f"Use Only Sequences with >= 2 Steps: {use_only_sequences_gte_2_steps}")
    print(f"Features: {hmm_config['features']}")
    