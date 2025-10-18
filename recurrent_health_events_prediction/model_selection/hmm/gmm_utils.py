import random
import plotly.graph_objects as go
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pomegranate.distributions import Normal, LogNormal, LogNormal, Exponential, Gamma, StudentT
from pomegranate.gmm import GeneralMixtureModel
from recurrent_health_events_prediction.model.model_types import DistributionType

def calculate_log_likelihood(model: GeneralMixtureModel, data: np.ndarray) -> float:
    """
    Calculate the log likelihood of the data given the model.
    """
    data = data.reshape(-1, 1)  # Ensure data is 2D
    log_likelihood = model.log_probability(data).sum().item()
    return log_likelihood

def get_number_params_of_distribution(distribution_type: DistributionType):
    """
    Returns the number of parameters for a given distribution type.
    """
    if distribution_type == DistributionType.GAUSSIAN or distribution_type == DistributionType.NORMAL:
        return 2  # mean and std
    elif distribution_type == DistributionType.LOG_NORMAL:
        return 2  # mean and std
    elif distribution_type == DistributionType.GAMMA:
        return 2  # shape and scale
    elif distribution_type == DistributionType.EXPONENTIAL:
        return 1  # rate
    elif distribution_type == DistributionType.STUDENT_T:
        return 2  # degrees of freedom and scale
    elif distribution_type == DistributionType.WEIBULL:
        return 2  # shape and scale
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def calculate_aic(model: GeneralMixtureModel, log_likelihood: float, distribution_type: DistributionType) -> float:
    """
    Compute AIC (Akaike Information Criterion) for the model on the given data.
    """
    num_params_per_component = get_number_params_of_distribution(distribution_type)
    num_components = len(model.distributions)
    k = num_params_per_component * num_components
    aic = 2 * k - 2 * log_likelihood
    return aic

def calculate_bic(model: GeneralMixtureModel, data: np.ndarray, log_likelihood: float, distribution_type: DistributionType) -> float:
    """
    Compute BIC (Bayesian Information Criterion) for the model on the given data.
    """
    data = data.reshape(-1, 1)  # Ensure data is 2D
    num_params_per_component = get_number_params_of_distribution(distribution_type)
    num_components = len(model.distributions)
    k = num_params_per_component * num_components
    n = data.shape[0]
    bic = np.log(n) * k - 2 * log_likelihood
    return bic

def plot_model_gmm_selection_results(metric: str, variable: str, distribution_type: DistributionType, num_params_list: list[int], num_components_list: list[int],
                                     mean_vales: list[float], std_values: Optional[list[float]] = None,
                                     show_plot=True):
    """
    Plots the model selection results for HMM based on the specified metric.
    Parameters:
    - metric: The metric to plot (e.g., 'bic', 'aic').
    - num_params_list: List of number of parameters for each model.
    - num_components_list: List of number of states for each model.
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
            "Number of Components: %{customdata}"
        ),
        name=metric,
        error_y=dict(
            type='data',
            array=std_values,
            visible=True,
            thickness=2,  # optional, makes lines a bit thicker
            width=6       # optional, makes cap wider
        ) if std_values is not None else None,
        customdata=num_components_list  # Pass number of states as custom data
    ))

    fig.update_layout(
        title=f"{metric} vs Number of Parameters - GMM for {variable} ({distribution_type.name.title()})",
        xaxis_title="Number of Parameters",
        yaxis_title=metric,
        template="plotly_white"
    )
    if show_plot:
        fig.show()
    return fig

def gmm_model_selection(data, max_components=5, var_name="Variable", distribution_type=DistributionType.NORMAL, dof=None):
    dist_map = {
        DistributionType.GAUSSIAN: Normal,
        DistributionType.LOG_NORMAL: LogNormal,
        DistributionType.GAMMA: Gamma,
        DistributionType.EXPONENTIAL: Exponential,
        DistributionType.STUDENT_T: StudentT
    }
    dist_class = dist_map.get(distribution_type, Normal)
    # Store metrics
    bics = []
    aics = []
    log_likelihoods = []
    # Store models
    models = []
    num_components_list = list(range(2, max_components+1))
    num_params_list = []

    if distribution_type == DistributionType.STUDENT_T and dof is None:
        dof = 5  # Default degrees of freedom for StudentT if not provided
    
    # Fit models with 1 up to max_components
    for n in num_components_list:
        distributions = [dist_class() if distribution_type != DistributionType.STUDENT_T else dist_class(dof) for _ in range(n)]
        mixture_model = GeneralMixtureModel(
            distributions=distributions
        )
        mixture_model.fit(data.reshape(-1, 1)) # Fit the model
        log_likelihood = calculate_log_likelihood(mixture_model, data)
        aic = calculate_aic(mixture_model, log_likelihood, distribution_type)
        bic = calculate_bic(mixture_model, data, log_likelihood, distribution_type)
        num_params = get_number_params_of_distribution(distribution_type) * n
        num_params_list.append(num_params)
        log_likelihoods.append(log_likelihood)
        bics.append(bic)
        aics.append(aic)
        models.append(mixture_model)
        print(f"GMM-{n} components: BIC={bic:.2f}, AIC={aic:.2f}")

    plot_model_gmm_selection_results("aic", var_name, distribution_type, num_params_list, num_components_list,
                                     aics, show_plot=True)
    plot_model_gmm_selection_results("bic", var_name, distribution_type, num_params_list, num_components_list,
                                     bics, show_plot=True)

    # Pick the best model (lowest BIC)
    best_idx = np.argmin(bics)
    best_model = models[best_idx]
    print(f"\nBest GMM: {num_components_list[best_idx]} components (BIC={bics[best_idx]:.2f})")
    
    # Plot empirical and GMM density
    x = np.linspace(data.min(), data.max(), 1000).reshape(-1,1)
    logpdf = best_model.log_probability(x)
    pdf = np.exp(logpdf)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, label='Empirical', fill=True, color='black', lw=2, alpha=0.25)
    plt.plot(x.flatten(), pdf, label=f'Best GMM ({num_components_list[best_idx]} components)', color='blue')
    plt.title(f"Best GMM Fit for {var_name}")
    plt.xlabel(var_name)
    plt.ylabel("Density")
    plt.legend()
    sns.despine()
    plt.show()
    
    return best_model

def get_gmm_params_df(model, distribution_type: DistributionType) -> pd.DataFrame:
    """
    Returns a DataFrame with the parameters for each component in a pomegranate GMM (new API).
    Supports StudentT, Normal (Gaussian), and Gamma components.
    """
    if distribution_type not in [DistributionType.STUDENT_T, DistributionType.NORMAL, DistributionType.GAMMA]:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    if distribution_type == DistributionType.STUDENT_T:
        columns = [
            'Component', 'Distribution', 'nu (dof)', 'Mean', 'Variance'
        ]
    elif distribution_type == DistributionType.NORMAL:
        columns = [
            'Component', 'Distribution', 'Mean', 'Variance'
        ]
    elif distribution_type == DistributionType.GAMMA:
        columns = [
            'Component', 'Distribution', 'Shape', 'Rate', 'Expected Value'
        ]

    data = []

    for i, dist in enumerate(model.distributions):
        # Default values for columns
        nu = mean = cov = shape = rate = expected = None

        if isinstance(dist, StudentT):
            nu = dist.dofs.numpy().item()
            mean = dist.means.numpy().item()
            cov = dist.covs.numpy().item()
            row = [
                i,
                type(dist).__name__,
                nu,
                mean,
                cov,
            ]
        elif isinstance(dist, Normal):
            mean = dist.means.numpy().item()
            cov = dist.covs.numpy().item()
            row = [
                i,
                type(dist).__name__,
                mean,
                cov,
            ]
        elif isinstance(dist, Gamma):
            shape = dist.shapes.numpy().item()
            rate = dist.rates.numpy().item()
            expected = shape / rate if rate != 0 else None
            row = [
                i,
                type(dist).__name__,
                shape,
                rate,
                expected,
            ]
        
        data.append(row)

    return pd.DataFrame(data, columns=columns)

