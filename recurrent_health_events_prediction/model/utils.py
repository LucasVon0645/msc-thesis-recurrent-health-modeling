from typing import Optional
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from pomegranate.distributions import Normal, Poisson, Categorical, Bernoulli, Gamma, Exponential, StudentT
from pomegranate.hmm import DenseHMM

from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from lifelines import LogNormalAFTFitter
import torch

def get_expected_value(distribution) -> np.ndarray:
    """
    Get the expected value of a distribution.
    
    Args:
        distribution: The distribution object (Normal, Poisson).    
    Returns:
        The expected value of the distribution.
    """
    if isinstance(distribution, Normal) or isinstance(distribution, StudentT):
        return distribution.means.detach().numpy()  # Mean
    elif isinstance(distribution, Poisson):
        return distribution.lambdas.detach().numpy()  # Lambda
    elif isinstance(distribution, Categorical):
        # For Categorical, we return the expected value as a weighted sum of categories
        probs = distribution.probs.detach().numpy()[0]
        categories = np.arange(len(probs))
        expected_value = np.sum(categories * probs)
        return expected_value
    elif isinstance(distribution, Bernoulli):
        probs = distribution.probs.detach().numpy()
        return probs[0].item()  # Probability of success
    elif isinstance(distribution, Gamma):
        return distribution.shapes.detach().numpy() / distribution.rates.detach().numpy() # Mean of Gamma is shape/rate
    elif isinstance(distribution, Exponential):
        return 1 / distribution.scales.detach().numpy()  # Mean of Exponential is 1/lambda
    else:
        raise ValueError(f"Unsupported distribution type: {type(distribution)}")

def normalize_probabilities(probs: np.ndarray):
    """
    Normalize a probability distribution.
    
    Args:
        probabilities: A numpy array of probabilities.
        
    Returns:
        A normalized numpy array where the sum is 1.
    """
    norm_probs = np.exp(probs) / np.sum(np.exp(probs))
    return norm_probs

def plot_survival_function_gbm_survival(surv_funcs, n=5):
    for i, fn in enumerate(surv_funcs[:n]):
        plt.step(fn.x, fn.y, where="post", label=f"Sample {i}")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_survival_function_plotly(surv_funcs: pd.DataFrame, n=5,
                                  true_events_durations_df: Optional[pd.DataFrame] = None,
                                  event_id_col: str = 'EVENT_ID',
                                  show_plot: bool = True,
                                  title: str = "Survival Functions"):
    """
    Plot survival functions from lifelines using plotly.
    
    Args:
        surv_funcs: A DataFrame containing survival functions.
        n: Number of samples to plot.
        true_events: Optional DataFrame containing true event occurrences.
        true_durations: Optional DataFrame containing true durations.
    """
    surv_funcs = surv_funcs.iloc[:, :n]
    true_events_durations_df = true_events_durations_df.iloc[:n, :] if true_events_durations_df is not None else None

    fig = px.line(
        surv_funcs,
        x=surv_funcs.index,
        y=surv_funcs.columns,
        labels={"index": "Time", "value": "Survival probability", "variable": event_id_col},
        title=title
    )
    fig.for_each_trace(lambda t: t.update(name=f"Surv. Curve Event {t.name}"))
    fig.update_layout(
        xaxis_title="Time (days)",
        yaxis_title="Survival Probability",
        legend_title="",
    )

    if true_events_durations_df is not None:
        for idx, row in true_events_durations_df.iterrows():
            x = round(row["true_duration"], 4)  # Time
            y = round(row["survival_prob_at_true_duration"], 4)  # Probability
            event_id = int(row[event_id_col])  # Event ID
            event = int(row["true_event"])  # Event indicator
            color = "black" if event else "gray"
            name = f"Event {event_id}: Obs" if event else f"Event {event_id}: Cens"
            fig.add_scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker=dict(size=8, color=color, symbol="circle"),
                name=name,
                hovertemplate=(
                    f"<b>Event ID:</b> {event_id}<br>"
                    f"<b>True Duration:</b> {x}<br>"
                    f"<b>Survival Probability:</b> {y}<br>"
                    f"<b>Event:</b> {event}<extra></extra>"
                ),
                showlegend=True,
            )
    if show_plot:
        fig.show()
    
    return fig

def plot_brier_score(brier_scores_df: pd.DataFrame, title: str = "Brier Scores", save_path: Optional[str] = None, show_plot = True):
    """
    Plot Brier scores from a DataFrame.
    
    Args:
        brier_scores_df: A DataFrame containing Brier scores.
        
    Returns:
        A plotly figure showing the Brier scores.
    """
    fig = px.line(
        brier_scores_df,
        x=brier_scores_df["time"],
        y=brier_scores_df["brier_score"],
        title=title,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Brier Score"
    )
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        # Save the figure as an HTML file
        fig.write_html(save_path)
    if show_plot:
        fig.show()

def plot_auc_roc_dynamic(auc_roc_dynamic_df: pd.DataFrame, title: str = "AUC ROC Dynamic Scores", save_path: Optional[str] = None, show_plot = True):
    """
    Plot AUC ROC dynamic scores from a DataFrame.
    
    Args:
        auc_roc_dynamic_df: A DataFrame containing AUC ROC dynamic scores.
        
    Returns:
        A plotly figure showing the AUC ROC dynamic scores.
    """
    fig = px.line(
        auc_roc_dynamic_df,
        x=auc_roc_dynamic_df["time"],
        y=auc_roc_dynamic_df["auc"],
        title=title,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="AUC ROC"
    )
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        # Save the figure as an HTML file
        fig.write_html(save_path)
    if show_plot:
        fig.show()

def plot_auc(y_pred, y_true, title: str = "AUC ROC Curve", save_path: Optional[str] = None, show_plot=True):
    """
    Plot AUC ROC curve.
    
    Args:
        y_pred: Predicted probabilities.
        y_true: True binary labels.
        title: Title of the plot.
        
    Returns:
        A plotly figure showing the AUC ROC curve.
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=title,
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        width=800,
        height=600
    )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Guessing')
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True
    )
    fig.add_annotation(
        text=f"AUC = {roc_auc:.2f}",
        xref="paper", yref="paper",
        x=0.95, y=0.05,
        showarrow=False,
        font=dict(size=12)
    )
    if save_path:
        if not save_path.endswith('.html'):
            save_path += '.html'
        # Save the figure as an HTML file
        fig.write_html(save_path)
    if show_plot:
        fig.show()

def get_feature_importance_df(model):
    """
    Get feature importance from a GradientBoostingSurvivalAnalysis model.
    
    Args:
        model: A trained GradientBoostingSurvivalAnalysis model.
        
    Returns:
        A DataFrame containing feature importances.
    """
    feature_importances = model.feature_importances_
    feature_names = model.feature_names_in_
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values(by='importance', ascending=False)

def plot_model_feature_importance(model, n=None, title: str = "Feature Importance", save_path: Optional[str] = None, show_plot=True):
    """
    Plot feature importance for a ensemble model.
    Args:
        model: A trained GradientBoostingSurvivalAnalysis model.
        n: Number of top features to plot.
        title: Title of the plot.
        save_path: Path to save the plot.
        show_plot: Whether to display the plot.
    Returns:
        A matplotlib figure showing feature importances.
    """
    feature_importance_df = get_feature_importance_df(model)
    if n is not None:
        feature_importance_df = feature_importance_df.head(n)

    # Plot feature importances
    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.title(title)
    if save_path:
        if not save_path.endswith('.png'):
            save_path += '.png'
        fig.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()
    return fig

def get_median_and_ci(surv_df: pd.DataFrame, ci: float = 0.95) -> pd.DataFrame:
    """
    Vectorized computation of median survival time and confidence interval bounds for each subject.
    
    Parameters:
        surv_df (pd.DataFrame): A DataFrame with survival probabilities.
                                Index = time points, Columns = subjects.
        ci (float): Confidence interval level (default = 0.95).
    
    Returns:
        pd.DataFrame: A DataFrame with index = subjects and columns = [event_id, median_survival_prob, lower_ci_XX, upper_ci_XX].
    """
    times = surv_df.index.values
    surv_np = surv_df.to_numpy()
    
    # Survival probability thresholds
    lower_prob = 1 - (1 - ci) / 2  # e.g., 0.975
    upper_prob = (1 - ci) / 2      # e.g., 0.025

    def first_time_le(prob_threshold):
        mask = surv_np <= prob_threshold
        # Replace False with np.nan to ignore in min
        time_matrix = np.where(mask, times[:, None], np.nan)
        return np.nanmin(time_matrix, axis=0)
    
    median_times = first_time_le(0.5)
    lower_ci_times = first_time_le(lower_prob)
    upper_ci_times = first_time_le(upper_prob)

    result_df = pd.DataFrame({
        'event_id': surv_df.columns,
        'median_survival_prob': median_times,
        f'lower_ci_{int(ci * 100)}': lower_ci_times,
        f'upper_ci_{int(ci * 100)}': upper_ci_times
    })
    
    return result_df

def lognormal_pdf_from_params(log_normal_aft_model: LogNormalAFTFitter, df: pd.DataFrame, t: Optional[float | np.ndarray] = None):
    """
    Compute the Log-Normal PDF for individuals in df using lifelines-style AFT parameters.
    
    Parameters:
        df (pd.DataFrame): Input covariates (same columns as model).
        log_normal_aft_model (LogNormalAFTFitter): Fitted LogNormal AFT model from lifelines.
        t (float | np.ndarray, optional): Time points to evaluate the PDF. If None, defaults to np.linspace(0.01, 100, 365).

    Returns:
        np.ndarray: PDF values (shape: [n_samples, len(t)])
    """
    if not isinstance(log_normal_aft_model, LogNormalAFTFitter):
        raise ValueError("log_normal_aft_model must be an instance of LogNormalAFTFitter.")
    
    if t is None:
        t = np.linspace(0.01, 100, 365)
    
    # Extract parameters from the fitted model
    param_series = log_normal_aft_model.params_

    # Extract mu_ coefficients
    mu_params = param_series.loc['mu_']
    intercept_mu = mu_params.get('Intercept', 0.0)
    mu_params = mu_params.drop('Intercept', errors='ignore')

    # Compute μ(x)
    mu = df[mu_params.index].dot(mu_params.values) + intercept_mu  # shape (n_samples,)

    # Extract σ from sigma_ Intercept
    sigma_log = param_series.loc['sigma_']['Intercept']
    sigma = np.exp(sigma_log)  # scalar

    # Prepare t
    t = np.atleast_1d(t)
    t = t.reshape(1, -1)             # (1, len(t))
    mu = mu.values.reshape(-1, 1)    # (n_samples, 1)

    # Log-normal PDF formula
    pdf = (1 / (t * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(t) - mu) ** 2) / (2 * sigma ** 2))
    return pdf  # shape (n_samples, len(t))

def compute_lognormal_mode(log_normal_aft_model: LogNormalAFTFitter, df: pd.DataFrame):
    """
    Compute the mode of the Log-Normal distribution for each individual in df 

    Args:
        df (pd.DataFrame): Covariate values.
        log_normal_aft_model (LogNormalAFTFitter): Fitted LogNormal AFT model from lifelines.

    Returns:
        pd.Series: Mode values.
    """
    if not isinstance(log_normal_aft_model, LogNormalAFTFitter):
        raise ValueError("log_normal_aft_model must be an instance of LogNormalAFTFitter.")
    
    # Extract parameters from the fitted model
    param_series = log_normal_aft_model.params_

    mu_params = param_series.loc["mu_"]
    intercept_mu = mu_params.get("Intercept", 0.0)
    mu_params = mu_params.drop("Intercept", errors="ignore")

    mu = df[mu_params.index].dot(mu_params.values) + intercept_mu
    sigma_log = param_series.loc["sigma_"]["Intercept"]
    sigma = np.exp(sigma_log)

    mode = np.exp(mu - sigma**2)
    return pd.Series(mode, index=df.index, name="mode")

def predict_single_obs_proba(model: DenseHMM, obs: list):
    """
    Predict the probabilities of each hidden state given a single observation.
    """
    emission_log_probs = []

    for dist in model.distributions:
        log_prob = dist.log_probability(torch.tensor([obs]))  # batch size 1
        emission_log_probs.append(log_prob.item())

    # Combine with start probabilities
    joint_log_probs = [
        start_log + emission_log
        for start_log, emission_log in zip(model.starts.tolist(), emission_log_probs)
    ]

    # Convert to normal probability space using softmax (log-sum-exp trick)
    joint_probs = torch.nn.functional.softmax(torch.tensor(joint_log_probs), dim=0).tolist()

    return joint_probs

def predict_most_likely_state_single_obs(model: DenseHMM, obs: list):
    """
    Predict the most likely hidden state given a single observation.
    """
    joint_probs = predict_single_obs_proba(model, obs)
    most_likely_state = np.argmax(joint_probs)
    return most_likely_state.item()

def load_model(model_path: str):
    """
    Load a model from a specified path.
    
    Args:
        model_path (str): Path to the model file.
        
    Returns:
        The loaded model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_single_partial_obs_proba(model: DenseHMM, current_obs_partial: list):
    """
    Predict the probabilities of each hidden state given a single observation with missing values.
    Uses log-sum-exp trick for numerical stability.
    """
    n_states = model.n_distributions

    # Emission likelihoods in log space
    emission_log_probs = []
    for state_idx in range(n_states):
        state_dist = model.distributions[state_idx]
        state_log_prob = 0.0
        for i, value in enumerate(current_obs_partial):
            if not pd.isna(value):
                state_log_prob += state_dist.distributions[i].log_probability(
                    torch.tensor([[value]])
                ).item()
        emission_log_probs.append(state_log_prob)

    # Combine with start probabilities in log space
    joint_log_probs = torch.tensor([
        start_log + emission_log
        for start_log, emission_log in zip(model.starts.tolist(), emission_log_probs)
    ])

    # Stable softmax using log-sum-exp
    max_log = joint_log_probs.max()
    shifted = joint_log_probs - max_log
    exp_shifted = torch.exp(shifted)
    probs = exp_shifted / exp_shifted.sum()

    return probs.tolist()

def predict_partial_obs_given_history_proba(
    model: DenseHMM, 
    past_sequence, 
    current_obs_partial,
    return_all_probs_all_steps=False
) -> list:
    """
    Estimate hidden state probabilities for a current observation with missing values,
    given a fully observed history. Uses softmax normalization for all steps.
    If return_all_probs_all_steps is True, returns probabilities for all past states as well.
    """
    n_states = model.n_distributions

    # 1. Get forward probabilities for past sequence (log space)
    with torch.no_grad():
        past_sequence_tensor = torch.tensor(past_sequence)
        if past_sequence_tensor.ndim == 2:
            past_sequence_tensor = past_sequence_tensor.unsqueeze(0)
        
        log_alpha = model.forward(past_sequence_tensor).squeeze(0)
    
    # Normalize log_alpha with softmax (row-wise)
    alpha_norm = torch.softmax(log_alpha, dim=1)  # shape: (n_obs, n_states)
    alpha_prev = alpha_norm[-1]  # Last time step's state probs

    # 2. Transition matrix (already in log space)
    log_transition = model.edges
    transition_matrix = torch.softmax(log_transition, dim=1)  # softmax across states

    # 3. Emission likelihoods in log space
    emission_log_probs = []
    for state_idx in range(n_states):
        state_dist = model.distributions[state_idx]
        state_log_prob = 0.0
        for i, value in enumerate(current_obs_partial):
            if not pd.isna(value):
                state_log_prob += state_dist.distributions[i].log_probability(
                    torch.tensor([[value]])
                ).item()
        emission_log_probs.append(state_log_prob)
    
    # Convert emission log-probs to probabilities using softmax
    emission_log_probs = torch.tensor(emission_log_probs)
    emission_probs = torch.softmax(emission_log_probs, dim=0)  # softmax across states

    # 4. Compute next alpha (probabilities)
    trans_probs = torch.matmul(alpha_prev, transition_matrix)
    alpha_next = trans_probs * emission_probs  # Element-wise multiplication

    # 5. Final normalization using softmax (for extra safety; could also just divide by sum)
    state_probs = alpha_next / (alpha_next.sum() + torch.finfo(torch.float32).eps)
    state_probs = state_probs.detach().numpy().tolist()

    if return_all_probs_all_steps:
        past_state_probs = alpha_norm.detach().numpy().tolist()
        past_state_probs.append(state_probs)  # Append current state probs to past
        return past_state_probs

    return state_probs  # Return only current state probabilities
