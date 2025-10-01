import os
from typing import Optional
import neptune
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# Define the color palette using Plotly's qualitative colors
plotly_colors = px.colors.qualitative.Plotly
sns.set_palette(plotly_colors)

from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from recurrent_health_events_prediction.model.NextEventPredictionModel import NextEventPredictionModel, NextEventSurvivalWrapper
from recurrent_health_events_prediction.model.utils import plot_auc, plot_brier_score
from recurrent_health_events_prediction.model.utils import plot_auc_roc_dynamic

from recurrent_health_events_prediction.model.explainers import explain_survival_model_prob, plot_waterfall

event_type_color_map = {
    "Event Ocurred": px.colors.qualitative.Plotly[1],
    "No Event": px.colors.qualitative.Plotly[0]
}

event_type_censoring_color_map = {
    "Event Ocurred": px.colors.qualitative.Plotly[1],
    "Censored": px.colors.qualitative.Plotly[0]
}

def set_observation_window(row, limit=30, event_col='READMISSION_EVENT'):
    if row['EVENT_DURATION'] > limit:
        row['EVENT_DURATION'] = limit
        row[event_col] = 0
    if row[event_col] == 0:
        row['EVENT_DURATION'] = limit
    return row

def train_test_split_survival_data(events_df: pd.DataFrame, 
                                   duration_col: str,
                                   event_col: str,
                                   q_bins: int = 4,
                                   additional_stratify_col: str = None,
                                   verbose=True) -> tuple:
    """
    Splits the dataset into training and testing sets with stratification based on event duration and censoring.

    Parameters:
    - events_df: DataFrame containing the events data.
    - duration_col: Column name for event duration.
    - event_col: Column name indicating if the event occurred.
    - q_bins: Number of bins to categorize the duration for stratification.
    - verbose: If True, prints the composition of train and test sets.

    Returns:
    - X_train: Training set features.
    - X_test: Testing set features.
    """

    # Create bins for stratification
    # Use quantiles to create bins with approximately equal-sized groups
    events_df['_duration_bins'] = pd.qcut(events_df[duration_col], q=q_bins, labels=False)
    if additional_stratify_col:
        # If an additional stratification column is provided, combine it with the duration bins
        events_df['_stratify_col'] = events_df['_duration_bins'].astype(str) + "_" + events_df[event_col].astype(str) + events_df[additional_stratify_col].astype(str)
    else:
        events_df['_stratify_col'] = events_df['_duration_bins'].astype(str) + "_" + events_df[event_col].astype(str)

    # Split the dataset with stratification based on duration bins
    X_train, X_test = train_test_split(
        events_df,
        test_size=0.2,  # 20% for testing
        stratify=events_df['_duration_bins'],  # Stratify by duration bins
        random_state=42
    )

    # Ensure no datapoint in X_test has a greater EVENT_DURATION than in X_train
    max_train_duration = X_train[duration_col].max()
    # Move datapoints in X_test with duration greater than max_train_duration to X_train
    out_of_range_test = X_test[X_test[duration_col] > max_train_duration]
    X_train = pd.concat([X_train, out_of_range_test], ignore_index=True)
    X_test = X_test[X_test[duration_col] <= max_train_duration]

    # Drop the duration_bins column after splitting
    X_train = X_train.drop(columns=['_duration_bins', '_stratify_col'])
    X_test = X_test.drop(columns=['_duration_bins', '_stratify_col'])

    if verbose:
        print("Train set event composition:\n", X_train[event_col].value_counts() / X_train.shape[0])
        print("Test set event composition:\n", X_test[event_col].value_counts() / X_test.shape[0])
        # Verify the mean duration in both splits
        print("Train average event duration:", X_train[duration_col].mean())
        print("Test average event duration:", X_test['EVENT_DURATION'].mean())
        if additional_stratify_col:
            print("Train set stratification column composition:\n", X_train[additional_stratify_col].value_counts() / X_train.shape[0])
            print("Test set stratification column composition:\n", X_test[additional_stratify_col].value_counts() / X_test.shape[0])

    return X_train, X_test

def build_strata_col(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    q_bins: int = 4,
    additional_stratify_col: str = None,
    strata_col_name: str = "_strata_col",
    duration_bins_col: str = "_duration_bins",
) -> pd.Series:
    """
    Returns a stratification column of a DataFrame for survival analysis CV splitting.

    Parameters:
    - df: DataFrame
    - duration_col: column with durations/times
    - event_col: event indicator (1=event, 0=censored)
    - q_bins: number of quantile bins (default=4)
    - additional_stratify_col: (optional) column to further stratify by (e.g., 'SEX')
    - strata_col_name: name for the output stratification column
    - duration_bins_col: name for the intermediate duration bins column

    Returns:
    - strata_col: Series containing the stratification information
    """

    df = df.copy()
    # Create duration bins
    df[duration_bins_col] = pd.qcut(df[duration_col], q=q_bins, labels=False)

    # Build strata_col string
    if additional_stratify_col:
        df[strata_col_name] = (
            df[duration_bins_col].astype(str) + "_" +
            df[event_col].astype(str) +
            df[additional_stratify_col].astype(str)
        )
    else:
        df[strata_col_name] = (
            df[duration_bins_col].astype(str) + "_" +
            df[event_col].astype(str)
        )
    strata_col = df[strata_col_name].copy()
    return strata_col

def train_next_event_survival_model_rand_search_cv(
        training_df: pd.DataFrame, model_config: dict,
        strata_col_s: pd.Series,
        param_grid: dict, n_iter=10, random_state=42, cv=5):
    """
    Perform training with hyperparameter tuning for the NextEventSurvivalWrapper model using RandomizedSearchCV.
    Args:
        training_df (pd.DataFrame): DataFrame containing the training data.
        model_config (dict): Configuration dictionary for the model, including features, event_col, duration_col, etc.
        strata_col_s (pd.Series): Series containing the stratification column.
        param_grid (dict): Dictionary containing hyperparameter grid for tuning.
        n_iter (int): Number of iterations for RandomizedSearchCV.
        random_state (int): Random state for reproducibility.
        cv (int): Number of cross-validation folds.
    Returns:
        None
    """
    
    feature_cols = model_config["features"]
    event_col = model_config.get('event_col')
    duration_col = model_config.get('duration_col')
    cluster_col = model_config.get('cluster_col', None)
    strata_col = model_config.get('strata_col', None)
    cols_order = feature_cols + [duration_col, event_col]
    
    if cluster_col is not None:
        cols_order = cols_order + [cluster_col]
        
    if strata_col is not None:
        cols_order = cols_order + [strata_col]
        
    X = training_df[cols_order]
    
    model = NextEventSurvivalWrapper(
            columns_order=cols_order,
            model_config=model_config
        )

    param_dist = {
        'model_params': list(ParameterSampler(param_grid, n_iter=n_iter, random_state=random_state))
    }
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=skf.split(X, strata_col_s),
        random_state=random_state,
        n_jobs=-1,
    )

    random_search.fit(X)

    return random_search

def evaluate_model(model: NextEventPredictionModel, X_train: pd.DataFrame, X_test: pd.DataFrame, evaluation_times: list, evaluation_set='test', save_plots=False, show_plot=True):
    """
    Show evaluation metrics for a given model.

    This function evaluates the model's performance on the test set using C-index, Brier score, and AUC-ROC.
    It also plots the Brier score and AUC-ROC curves.

    Parameters:
        model (NextEventPredictionModel): The model to evaluate.
        X_train (pd.DataFrame): Training dataset.
        X_test (pd.DataFrame): Test dataset.
        evaluation_times (list): List of times at which to evaluate the model's performance.

    Args:
        model (NextEventPredictionModel): The model to evaluate.
        X_train (pd.DataFrame): The training dataset.
        X_test (pd.DataFrame): The test dataset.
        evaluation_times (list): List of times at which to evaluate the model's performance.
    """
    model_name = model.model_name
    plots_dir = model.get_model_dir()
    plot_file_path = None

    c_index = model.evaluate_c_index(X_test, save_metric=True, evaluation_set=evaluation_set)
    print(f"Test C-index for {model_name}: {c_index:.4f}")

    brier_scores_df = model.evaluate_brier_score(X_train, X_test, evaluation_times=evaluation_times, save_metric=True, evaluation_set=evaluation_set)
    if save_plots:
        filename = f"brier_scores_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, filename)
    plot_brier_score(brier_scores_df, title=f"Brier Score - Test Set<br>{model_name}", save_path=plot_file_path, show_plot=show_plot)

    auc_df, mean_auc = model.evaluate_cumulative_dynamic_auc(X_train, X_test, evaluation_times=evaluation_times, save_metric=True, evaluation_set=evaluation_set)
    if save_plots:
        filename = f"dynamic_auc_roc_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, filename)
    plot_auc_roc_dynamic(auc_df, title=f"Dynamic AUC-ROC - Test Set<br>{model_name}", save_path=plot_file_path, show_plot=show_plot)

    print(f"Mean AUC-ROC for {model_name}: {mean_auc:.4f}")

def evaluate_model_around_true_event_durations(model: NextEventPredictionModel, X_input, nbins=50, opacity=0.7, layout_dict=None):
    """
    Evaluates a survival model and plots relevant metrics.

    Parameters:
    - model: The survival model instance (e.g., Kaplan-Meier, Cox Proportional Hazard).
    - X_train: Training dataset.
    - X_test: Test dataset.
    - duration_col: Column name for event duration.
    - event_col: Column name for event occurrence.
    - evaluation_times: Array of evaluation times for dynamic AUC-ROC.
    - nbins: Number of bins for survival probability distribution plot.
    - opacity: Opacity for the plot.

    Returns:
    - auc_roc_at_true_duration: AUC-ROC at true duration.
    """
    # Evaluate AUC-ROC at true duration
    prob_df, auc_roc_at_true_duration = model.evaluate_model_at_true_duration(X_input)
    print(f"{model.model_type} Model AUC-ROC around true duration: {auc_roc_at_true_duration:.4f}")

    title = f"Event Probability Distribution around True Event Durations"

    if layout_dict is None:
        layout_dict = {
		"template": "plotly_white",
		"width": 1000,
		"height": 300,
		"margin": dict(l=50, r=50, t=50, b=50),
		"title": title,
		"legend_title": "True Event Status",
		"yaxis_title": "Count",
	}
    else:
        layout_dict["title"] = title

    prob_df['event_status'] = prob_df['true_event'].map({0: "Censored", 1: "Event Occurred"})

    # Plot survival probability distribution
    plot_probability_distribution(
        prob_df,
        prob_col="prob_event_happened_before_true_duration_plus_one",
        event_col="event_status",
        color_map=event_type_censoring_color_map,
        nbins=nbins,
        opacity=opacity,
        layout_dict=layout_dict
    )

    fig = px.pie(prob_df,
       names='event_status',
       title='Distribution of Event Types',
       labels={'event_status': 'Event Status'},
       color='event_status',
       color_discrete_map=event_type_censoring_color_map,
       width=800,
       height=400)

    # Display the figure
    fig.show()

    return prob_df, auc_roc_at_true_duration

def plot_calibration_curve(labels, pred_prob, n_bins=5, title = "Calibration Curve", save_path: Optional[str] = None, show_plot=True):
    """
    Plots a calibration curve using Plotly.

    Args:
        labels (array-like): True binary labels (0 or 1).
        pred_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to use for calibration curve.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(labels, pred_prob, n_bins=n_bins)

    # Create the plot
    fig = go.Figure()

    # Add calibration curve
    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Calibration Curve'
    ))

    # Add perfectly calibrated line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfectly Calibrated',
        line=dict(dash='dash')
    ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Events Happened)",
        legend_title="Legend",
        template="plotly_white"
    )
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        # Save the figure as an HTML file
        fig.write_html(save_path)
    if show_plot:
        fig.show()

def plot_probability_distribution(prob_df, prob_col,
                                           event_col,
                                           layout_dict,
                                           color_map: Optional[dict] = None,
                                           show_plot = True,
                                           nbins=50, opacity=0.7, save_path: Optional[str] = None):
    """
    Plots the distribution of survival probabilities by true event.

    Args:
        prob_df (pd.DataFrame): DataFrame containing survival probabilities and true event information.
        prob_col (str): Column name for survival probabilities.
        event_col (str): Column name for true event categories.
        nbins (int): Number of bins for the histogram.
        opacity (float): Opacity of the histogram bars.

    Returns:
        None: Displays the plot.
    """
    fig = px.histogram(prob_df, x=prob_col, color=event_col, barmode="overlay", nbins=nbins, opacity=opacity, color_discrete_map=color_map)
    fig.update_layout(layout_dict)
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        # Save the figure as an HTML file
        fig.write_html(save_path)
    if show_plot:
        fig.show()

def evaluate_model_around_specific_time(model: NextEventPredictionModel,
                                        X_input, evaluation_time, save_plots=False, show_plot = True,
                                        nbins=50, opacity=0.7, layout_dict=None, evaluation_set='test', event_name="readmission"):
    """ Evaluates a survival model and plots relevant metrics around a specific time.

    Parameters:
    - model: The survival model instance (e.g., Kaplan-Meier, Cox Proportional Hazard).
    - X_input: Input dataset for evaluation.
    - evaluation_time: Specific time point to evaluate the model.
    - nbins: Number of bins for survival probability distribution plot.
    - opacity: Opacity for the plot.
    - layout_dict: Layout dictionary for the plot.

    Returns:
    - prob_df: DataFrame containing survival probabilities at the specified time.
    """
    model_name = model.model_name
    plot_file_path = None
    plots_dir = model.get_model_dir()

    # Evaluate AUC-ROC at specific time
    prob_df, auc_roc_at_t = model.evaluate_model_at_time_t(X_input, evaluation_time, save_metric=True, evaluation_set=evaluation_set)

    labels = prob_df['event_at_t']
    scores = prob_df['prob_event_happened_before_t']
    if save_plots:
        plot_filename = f"auc_roc_at_t_{evaluation_time}_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, plot_filename)
    title = f"AUC ROC for {event_name.title()} Probability before {evaluation_time} Days<br>{model_name}"
    plot_auc(scores, labels, show_plot=show_plot, title=title, save_path=plot_file_path)

    title = f"{event_name.title()} Probability Distribution before {evaluation_time} Days<br>{model_name}"

    if layout_dict is None:
        layout_dict = {
		"width": 800,
		"height": 300,
		"margin": dict(l=50, r=50, t=50, b=50),
		"title": title,
		"legend_title": "True Event Status",
		"yaxis_title": "Count",
        "xaxis_title": f"Probability of {event_name.title()} before {evaluation_time} Days",
	}
    else:
        layout_dict["title"] = title

    prob_df['event_status'] = prob_df['event_at_t'].map({0: "No Event", 1: "Event Occurred"})

    if save_plots:
        plot_filename = f"probability_dist_at_t_{evaluation_time}_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, plot_filename)

    # Plot survival probability distribution
    plot_probability_distribution(
        prob_df,
        prob_col="prob_event_happened_before_t",
        event_col="event_status",
        nbins=nbins,
        opacity=opacity,
        color_map=event_type_color_map,
        layout_dict=layout_dict,
        save_path=plot_file_path,
        show_plot=show_plot
    )

    # Plot calibration curve
    if save_plots:
        filename = f"calibration_curve_at_t_{evaluation_time}_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, filename)
    title = f"Calibration Curve at {evaluation_time} Days ({event_name.title()} Event - {model_name})"
    plot_calibration_curve(prob_df['event_at_t'], scores, title=title, save_path=plot_file_path, show_plot=show_plot)

    # Create a pie chart for event distribution
    fig = px.pie(prob_df,
       names='event_status',
       title=f'Distribution of {event_name.title()} Event Types {evaluation_time} Days',
       labels={'event_status': 'Event Status'},
       color='event_status',
       color_discrete_map=event_type_color_map,
       width=800,
       height=400)
    if save_plots:
        filename = f"event_distribution_at_t_{evaluation_time}_{evaluation_set}_set.html"
        plot_file_path = os.path.join(plots_dir, filename)
        fig.write_html(plot_file_path)
    if show_plot:
        fig.show()

    return prob_df, auc_roc_at_t

def plot_confusion_matrix(labels, scores, threshold: float = 0.5):
    # Compute confusion matrix
    cm = confusion_matrix(labels, scores > threshold)  # Assuming threshold of 0.5 for binary classification

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Readmission", "Readmission"])
    disp.plot(cmap="Blues")

def explain_event(model: NextEventPredictionModel, X_train, X_test, original_data_df, event_id, t=30, max_display=10):
    """
    Explains the survival model probabilities for a specific event_id.

    Parameters:
        model: The trained survival model.
        X_train: Training dataset used for fitting the model.
        X_test: Test dataset containing the event to explain.
        original_data_df: Original DataFrame containing all events data.
        event_id: The unique identifier for the event to explain.
        features: List of feature names used in the model.
        t: Time point at which to evaluate the SHAP values (default is 30).
        max_display: Maximum number of features to display in the waterfall plot.

    Returns:
        shap_values: SHAP values for the event.
        explainer: SHAP explainer object.
    """
    event_preprocessed_df = X_test[X_test['HADM_ID'] == event_id]
    event_df = original_data_df[original_data_df['HADM_ID'] == event_id]

    features = model.feature_names_in_

    shap_values, explainer = explain_survival_model_prob(
        model, 
        X_train[features].sample(20), 
        event_preprocessed_df[features], 
        t=t
    )

    plot_waterfall(explainer, event_df[features],
                   shap_values[0],
                   feature_names=features, max_display=10)

    return shap_values, explainer

def plot_shap_distributions(explanation_df, title="SHAP Value Distributions", nbins=20):
    """
    Plots the distribution of SHAP values for features in the given DataFrame.

    Parameters:
        explanation_df (pd.DataFrame): DataFrame containing SHAP values for features.

    Returns:
        None: Displays the plot.
    """
    # Extract SHAP columns
    shap_columns = [col for col in explanation_df.columns if col.endswith('_SHAP')]

    # Determine the range for x-axis based on the SHAP values
    x_min = explanation_df[shap_columns].min().min()
    x_max = explanation_df[shap_columns].max().max()

    # Create subplots with multiple rows
    fig = make_subplots(rows=len(shap_columns), cols=1, vertical_spacing=0.1,
                        subplot_titles=shap_columns)

    # Add traces for each SHAP column
    for i, col in enumerate(shap_columns):
        fig.add_trace(
            go.Histogram(x=explanation_df[col], name=col, marker=dict(color='blue'), nbinsx=nbins),
            row=i + 1, col=1
        )

    # Update layout
    fig.update_layout(
        title=title,
        height=300 * len(shap_columns),  # Adjust height based on the number of rows
        width=1000,
        showlegend=False
    )

    # Set the same range for all x-axes
    for i in range(1, len(shap_columns) + 1):
        fig.update_xaxes(range=[x_min, x_max], row=i, col=1)

    fig.update_yaxes(title_text="Count")
    fig.show()

def plot_bad_good_predictions_event_ocurring(events_df: pd.DataFrame, prob_df: pd.DataFrame, prob_col: str,
                             features: list, true_event_col: str = 'true_event', event_id_col: str = 'EVENT_ID',
                            good_survival_threshold=0.5, bad_survival_threshold=0.7):
    """
    Plots scatter plots of features against event duration for good and bad predictions.

    Parameters:
    - events_df: DataFrame containing the last events data.
    - prob_df: DataFrame containing survival probabilities at true event durations.
    - features: List of feature names to plot.

    Returns:
    - None: Displays the scatter plots.
    """
    
    # Filter good and bad predictions based on survival probabilities
    bad_pred_df = prob_df[(prob_df[prob_col] > bad_survival_threshold) &
                        (prob_df[true_event_col] == 1)]
    good_pred_df = prob_df[(prob_df[prob_col] < good_survival_threshold) &
                        (prob_df[true_event_col] == 1)]

    bad_event_ids = bad_pred_df[event_id_col].unique()
    good_event_ids = good_pred_df[event_id_col].unique()

    bad_events_df =events_df[events_df[event_id_col].isin(bad_event_ids)].copy()
    bad_events_df = bad_events_df.merge(bad_pred_df, left_on=event_id_col, right_on=event_id_col, how='left')
    good_events_df = events_df[events_df[event_id_col].isin(good_event_ids)].copy()
    good_events_df = good_events_df.merge(good_pred_df, left_on=event_id_col, right_on=event_id_col, how='left')

    bad_events_df["Label"] = "Bad Prediction"
    good_events_df["Label"] = "Good Prediction"

    events_concatenated_df = pd.concat([bad_events_df, good_events_df], ignore_index=True)

    # Define plot settings
    num_features = len(features)
    max_cols = 3
    num_rows = (num_features + max_cols - 1) // max_cols

    fig = make_subplots(
        rows=num_rows, 
        cols=max_cols,
        subplot_titles=features
    )

    # Iterate through features
    for i, feature in enumerate(features):
        row = (i // max_cols) + 1
        col = (i % max_cols) + 1

        # Separate data by label
        good_df = events_concatenated_df[events_concatenated_df['Label'] == 'Good Prediction']
        bad_df = events_concatenated_df[events_concatenated_df['Label'] == 'Bad Prediction']

        # Add Good Prediction trace
        fig.add_trace(
            go.Scatter(
                x=good_df[feature],
                y=good_df['EVENT_DURATION'],
                mode='markers',
                name='Good Prediction' if i == 0 else '',
                marker=dict(color='green'),
                showlegend=(i == 0),
                text=good_df[prob_col],
                hovertemplate=(
                    f"{feature}:" + " %{x}<br>" +
                    "Event Duration: %{y}<br>" +
                    "Prob: %{text}<extra></extra>"
                ),
            ),
            row=row, col=col
        )

        # Add Bad Prediction trace
        fig.add_trace(
            go.Scatter(
                x=bad_df[feature],
                y=bad_df['EVENT_DURATION'],
                mode='markers',
                name='Bad Prediction' if i == 0 else '',
                text=bad_df[prob_col],
                hovertemplate=(
                    f"{feature}:" +" %{x}<br>" +
                    "Event Duration: %{y}<br>" +
                    "Prob: %{text}<extra></extra>"
                ),
                marker=dict(color='red'),
                showlegend=(i == 0),
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text="Event Duration", row=row, col=col)

    # Layout
    fig.update_layout(
        title="Scatter Plots of Features vs Event Duration",
        height=300 * num_rows,
        width=1200,
        legend=dict(
            title="Prediction Labels",
            itemsizing="constant",
            font=dict(size=12),
        )
    )

    fig.show()

def save_coef_lifelines_plot(ax, model):
    model_name = model.model_name
    ax.set_title(f"{model_name} Coefficients")
    os.makedirs(model.get_model_dir(), exist_ok=True)
    # Save the coefficients plot
    fig = ax.figure
    fig.savefig(f"{model.get_model_dir()}/lifelines_coefficients_plot.png", bbox_inches='tight')