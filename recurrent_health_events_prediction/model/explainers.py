import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from recurrent_health_events_prediction.model.NextEventPredictionModel import NextEventPredictionModel
from recurrent_health_events_prediction.model.model_types import SurvivalModelType

def make_predict_surv_prob_at_t(model: NextEventPredictionModel, t: float):
    def predict_surv_prob_at_t(X_input):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=model.feature_names_in_)
        
        surv_df = model.predict_survival(X_input, times=[t])
        return surv_df.loc[t].values  # 1D array of survival probs at time t
    return predict_surv_prob_at_t

def explain_survival_model_prob(model: NextEventPredictionModel, X_train, X_explain, t: float):
    predict_fn = make_predict_surv_prob_at_t(model, t)
    if model.model_type == SurvivalModelType.GBM:
        # For GBM survival models, we need to ensure the input is in the right format
        explainer = shap.TreeExplainer(predict_fn, X_train)
    else:
        explainer = shap.KernelExplainer(predict_fn, X_train)

    shap_values = explainer.shap_values(X_explain)

    return shap_values, explainer

def plot_waterfall(explainer, X_explain, shap_values: np.ndarray, feature_names=None, max_display=10):
    """
    Plot a SHAP waterfall plot for the given SHAP values.
    
    Args:
        shap_values: SHAP values to plot of a single instance.
        feature_names: Optional list of feature names.
        max_display: Maximum number of features to display.
        
    Returns:
        A plotly figure showing the SHAP waterfall plot.
    """

    if len(X_explain) != 1:
        raise ValueError("X_explain must contain exactly one row for waterfall plot.")
    if shap_values.ndim != 1:
        raise ValueError("shap_values must be a 1D array for waterfall plot.")

    shap.plots.waterfall(shap.Explanation(values=shap_values,
                                      base_values=explainer.expected_value,
                                      data=X_explain.iloc[0],
                                      feature_names=feature_names),
                                      max_display=max_display)

def plot_survival_shap_summary(model: NextEventPredictionModel, X_train, X_explain, t=30, title=None):
    """
    Generate and return a SHAP summary plot figure for survival probability at time `t`.
    
    Parameters:
    - model: trained NextEventPredictionModel
    - X_train: background data (usually training data)
    - X_explain: data to explain (e.g., test data or subset of training)
    - t: float, time at which to compute survival probabilities
    - title: optional plot title
    
    Returns:
    - fig: matplotlib.figure.Figure
    """
    feature_names = model.feature_names_in_
    # Ensure only selected features are used
    X_train_subset = X_train[feature_names].copy()
    X_explain_subset = X_explain[feature_names].copy()
    
    # Get SHAP values
    shap_values, _ = explain_survival_model_prob(
        model, X_train_subset, X_explain_subset, t=t
    )
    
    # Handle binary/multiclass shape
    if shap_values.ndim == 3:
        shap_values_to_plot = shap_values[:, :, 1]  # Class 1 survival
    else:
        shap_values_to_plot = shap_values

    # Create figure for SHAP plot
    fig = plt.figure()
    plt.title(title or f"SHAP Summary Plot (t={t})", pad=20)
    shap.summary_plot(shap_values_to_plot, 
                      X_explain_subset.astype(float),
                      feature_names=feature_names,
                      show=False)  # Don't auto-show so we can return fig
    return fig
