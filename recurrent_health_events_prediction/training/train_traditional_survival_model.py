import contextlib
import io
from typing import Optional
import neptune
import pandas as pd
import numpy as np
import os

from sklearn.discriminant_analysis import StandardScaler

from recurrent_health_events_prediction.model.NextEventPredictionModel import (
    NextEventPredictionModel,
    NextEventSurvivalWrapper,
)
from recurrent_health_events_prediction.model.explainers import (
    plot_survival_shap_summary,
)
from recurrent_health_events_prediction.model.model_types import SurvivalModelType
from recurrent_health_events_prediction.model.utils import plot_model_feature_importance
from recurrent_health_events_prediction.preprocessing.utils import remap_discharge_location, remap_mimic_races
from recurrent_health_events_prediction.training.utils import apply_train_test_split_file_survival, preprocess_features_to_one_hot_encode, summarize_search_results
from recurrent_health_events_prediction.training.utils_survival import (
    evaluate_model,
    evaluate_model_around_specific_time,
    save_coef_lifelines_plot,
    train_next_event_survival_model_rand_search_cv,
    train_test_split_survival_data,
    build_strata_col,
)
from recurrent_health_events_prediction.training.utils_traditional_classifier import impute_missing_features
from recurrent_health_events_prediction.utils.general_utils import check_if_file_exists, import_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_model_config_to_neptune,
    add_plot_to_neptune_run,
    add_plotly_plots_to_neptune_run,
    export_neptune_token_from_file,
    initialize_neptune_run,
    track_file_in_neptune,
    upload_model_to_neptune,
    upload_training_data_to_neptune,
)


def add_survival_training_data_stats_to_neptune(neptune_run, X, event_col, duration_col, neptune_path='training_data'):
    neptune_run[f"{neptune_path}/num_samples"] = len(X)
    neptune_run[f"{neptune_path}/num_features"] = X.shape[1] - 3  # Exclude duration and event columns
    neptune_run[f"{neptune_path}/censoring_distribution"] = X[event_col].value_counts(normalize=True).to_dict()
    neptune_run[f"{neptune_path}/duration_stats"] = {
        'mean': X[duration_col].mean(),
        'std': X[duration_col].std(),
        'min': X[duration_col].min(),
        'max': X[duration_col].max()
    }

def remap_strata_col(X, model_config: dict):
    strata_remap = model_config.get('strata_remap')
    strata_col = model_config.get('strata_col')
    print(f"Remapping strata column: {strata_col} with remap: {strata_remap}")
    if strata_remap is not None and strata_col is not None:
        X[strata_col] = X[strata_col].map(strata_remap)
    return X

def save_pred_output_files_to_csv(model: NextEventPredictionModel, X_test, main_evaluation_time: int, neptune_run: Optional[neptune.Run] = None):
    # Predict
    event_probs_at_t_df = model.predict_events_at_t(X_test, t=main_evaluation_time)
    median_survival_times_df = model.predict_median_with_ci_bounds(X_test, ci=0.95)
    if model.model_type == SurvivalModelType.COX_PH:
        partial_hazard_df = model.pred_partial_hazard(X_test)

    # Define file paths
    output_folder = model.get_model_dir()
    event_probs_at_t_filepath = os.path.join(output_folder, f"event_probabilities_at_{main_evaluation_time}.csv")
    median_survival_times_filepath = os.path.join(output_folder, "median_survival_times.csv")
    partial_hazards_filepath = os.path.join(output_folder, "partial_hazards.csv")

    # Save to CSV
    event_probs_at_t_df.to_csv(event_probs_at_t_filepath, index=False)
    median_survival_times_df.to_csv(median_survival_times_filepath, index=False)
    if model.model_type == SurvivalModelType.COX_PH:
        partial_hazard_df.to_csv(partial_hazards_filepath, index=False)

    # Upload to Neptune if run is provided
    if neptune_run is not None:
        neptune_run[f"prediction_outputs_test_set/event_probabilities_at_{main_evaluation_time}"].upload(event_probs_at_t_filepath)
        neptune_run["prediction_outputs_test_set/median_survival_times"].upload(median_survival_times_filepath)
        if model.model_type == SurvivalModelType.COX_PH:
            neptune_run["prediction_outputs_test_set/partial_hazards"].upload(partial_hazards_filepath)

def prepare_data(
    training_df: pd.DataFrame,
    scale_features,
    duration_col,
    event_col,
    event_id_col,
    q_bins=1,
    features_not_to_scale=None,
    use_fixed_train_test_split=False,
    split_csv_path=None,
    
):
    if use_fixed_train_test_split:
        if split_csv_path is None or not os.path.exists(split_csv_path):
            raise ValueError(
                "split_csv_path must be provided and exist when use_fixed_train_test_split is True."
            )
        print(f"Using fixed train-test split from {split_csv_path}")
        X_train, X_test, _, _ = apply_train_test_split_file_survival(training_df, split_csv_path, event_id_col)
    else:
        X_train, X_test = train_test_split_survival_data(
            training_df, duration_col, event_col, q_bins
        )

    if not scale_features:
        return X_train, X_test

    if features_not_to_scale is not None:
        # Identify features to scale
        features_to_scale = [
            col for col in X_train.columns if col not in features_not_to_scale
        ]
    else:
        features_to_scale = X_train.columns.tolist()

    features_to_scale.remove(duration_col)
    features_to_scale.remove(event_col)
    features_to_scale.remove(event_id_col)

    print(f"Scaling features: {features_to_scale}")

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    scaler = StandardScaler()

    X_train_scaled[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    return X_train_scaled, X_test_scaled

def plot_surv_curves_examples(
    model: NextEventPredictionModel, X, duration_col, event_col, neptune_run=None
):
    fig = model.plot_survival_function(
        X.iloc[:-5],
        duration_col=duration_col,
        event_col=event_col,
        show_plot=False,
        title=f"Survival Function Examples<br>{model_name}",
    )
    filepath = os.path.join(model.get_model_dir(), "survival_function_examples.html")
    fig.write_html(filepath)
    if neptune_run:
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig,
            filename="survival_function_plot.html",
            filepath="plots/evaluation/survival_function_examples",
        )

def build_strata_col_for_cv(X_train, duration_col, event_col, q_bins=1):
    strata_col = build_strata_col(X_train, duration_col=duration_col, event_col=event_col, q_bins=q_bins)
    return strata_col

def define_grid_search_params_cv(model_type):
    if model_type == SurvivalModelType.GBM.value:
        param_grid_gbm_survival = {
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5, 6, 7, 8],
        }
        return param_grid_gbm_survival

    param_lifelines_grid = {
        "penalizer": np.linspace(0.01, 0.4, 20),
        "l1_ratio": np.linspace(0.01, 0.4, 20),
    }

    return param_lifelines_grid

def add_proportional_cph_to_neptune(neptune_run, X, model):
    # 2. Capture stdout of check_assumptions
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.model.check_assumptions(
            X,
            p_value_threshold=0.05
        )

    # 3. Get the full text output
    check_assumptions_output = buffer.getvalue()

    # 4. Log the output to Neptune
    neptune_run[f"training/check_assumptions_result"] = check_assumptions_output

def upload_evaluation_outputs_to_neptune(neptune_run, folder_path, fixed_evaluation_time=30):
    """
    Uploads model outputs to the appropriate locations in a Neptune run.

    Parameters:
    - run: Neptune run object
    - folder_path: Path to the folder where all output files are saved
    """
    # Map of files to their Neptune destinations
    file_structure = {
        "plots/evaluation/multiple_times": [
            "brier_scores_test_set.html",
            "dynamic_auc_roc_test_set.html"
        ],
        "plots/evaluation/fixed_time": [
            f"probability_dist_at_t_{fixed_evaluation_time}_test_set.html",
            f"calibration_curve_at_t_{fixed_evaluation_time}_test_set.html",
            f"event_distribution_at_t_{fixed_evaluation_time}_test_set.html",
            f"auc_roc_at_t_{fixed_evaluation_time}_test_set.html"
        ]
    }

    for destination, files in file_structure.items():
        for filename in files:
            full_path = os.path.join(folder_path, filename)
            if os.path.exists(full_path):
                neptune_run[destination + "/" + filename].upload(full_path)
            else:
                print(f"⚠️ File not found: {filename}")

def upload_params_outputs_to_neptune(neptune_run, folder_path):
    """
    Uploads model parameters output files to the appropriate locations in a Neptune run.

    Parameters:
    - run: Neptune run object
    - folder_path: Path to the folder where all output files are saved
    """

    files = ["model_params.txt", "model_summary.csv", "lifelines_coefficients_plot.png"]

    for filename in files:
        full_path = os.path.join(folder_path, filename)
        if os.path.exists(full_path):
            neptune_run[filename].upload(full_path)
        else:
            print(f"⚠️ File not found: {filename}")

def plot_feature_importance_survival_model(
    model: NextEventPredictionModel,
    model_name: str,
    X,
    evaluation_time: int = 30,
    plot_shap=True,
    neptune_run=None,
    neptune_path="feature_importance",
    random_state=42,
):
    model_type = model.model_type
    model_dir = model.get_model_dir()
    if model_type == SurvivalModelType.GBM:
        title = f"{model_name} Feature Importances"
        fig = plot_model_feature_importance(model.model, title=title, show_plot=False)
        filename = f"{model_name.lower().replace(' ', '_')}_feat_import.png"
        fig.savefig(os.path.join(model_dir, filename), bbox_inches="tight")
        if neptune_run:
            add_plot_to_neptune_run(neptune_run, filename, fig, neptune_path)

    elif plot_shap and model_type in [
        SurvivalModelType.COX_PH,
        SurvivalModelType.COX_PH_SE,
        SurvivalModelType.WEIBULL_AFT,
    ]:
        X_background = X.iloc[min(len(X), 500)]
        X_explain = X.sample(
            min(len(X), 1000), random_state=random_state
        )  # Sample for SHAP to speed up computation
        print(f"Calculating SHAP values for {len(X_explain)} samples...")
        title = (
            f"{model_name} SHAP Feature Importance {evaluation_time} Days - {len(X_explain)} training samples"
        )
        fig = plot_survival_shap_summary(
            model, X_background, X_explain, title=title, t=evaluation_time
        )
        filename = f"{model_name.lower().replace(' ', '_')}_shap_feat_import.png"
        fig.savefig(os.path.join(model_dir, filename), bbox_inches="tight")
        if neptune_run:
            add_plot_to_neptune_run(neptune_run, filename, fig, neptune_path)

def train_traditional_survival_model(training_df: pd.DataFrame,
                                 model_config: dict,
                                 random_state=42,
                                 cv_n_folds=5,
                                 cv_n_iter=25,
                                 q_bins=1,
                                 plot_shap=True,
                                 neptune_run=None,
                                 missing_features=None,
                                 use_fixed_train_test_split=False,
                                 split_csv_path=None):

    if neptune_run:
        add_model_config_to_neptune(neptune_run, model_config)
        neptune_run["random_state"] = random_state

    evaluation_times = model_config['evaluation_times']
    main_evaluation_time = model_config['main_evaluation_time']
    scale_features = model_config['scale_features']
    features_not_to_scale = model_config['features_not_to_scale']
    duration_col = model_config['duration_col']
    event_col = model_config['event_col']
    event_id_col = model_config['event_id_col']
    event_name = model_config.get('event_name', 'readmission')
    features_cols = model_config['features']
    strata_col = model_config.get('strata_col', None)
    cluster_col = model_config.get('cluster_col', None)
    model_name = model_config['model_name']
    model_type = model_config['model_type']

    print("Training survival model...")
    print("Model name: ", model_name)
    print("Model type: ", model_type)
    print("Using features: ", features_cols)
    print("Event duration column: ", duration_col)
    print("Event column: ", event_col)
    print("Event name: ", event_name)
    print("Strata column: ", strata_col if strata_col else "None")
    print("Event ID column: ", event_id_col)
    print("Cluster column: ", cluster_col if cluster_col else "None")
    print("Evaluation times: ", evaluation_times)
    print("Main evaluation time: ", main_evaluation_time)

    # Add a random feature for reference
    np.random.seed(random_state)  # Ensure reproducibility
    training_df['RANDOM_FEATURE'] = np.random.rand(len(training_df))
    cols = [event_id_col] + features_cols + [duration_col, event_col]
    if strata_col:
        cols.append(strata_col)
    if cluster_col:
        cols.append(cluster_col)

    X = training_df[cols]

    if neptune_run:
        add_survival_training_data_stats_to_neptune(neptune_run, X, event_col, duration_col, "training_data")

    X_train, X_test = prepare_data(
        X,
        scale_features,
        duration_col,
        event_col,
        event_id_col,
        features_not_to_scale=features_not_to_scale,
        q_bins=q_bins,
        use_fixed_train_test_split=use_fixed_train_test_split,
        split_csv_path=split_csv_path,
    )

    if missing_features:
        X_train, X_test = impute_missing_features(X_train, X_test, missing_features)

    strata_col_cv = build_strata_col_for_cv(X_train, duration_col, event_col, q_bins=q_bins)

    param_grid = define_grid_search_params_cv(model_type)

    print(f"Training {model_name} with hyperparameter search...")

    random_search = train_next_event_survival_model_rand_search_cv(
        training_df=X_train,
        model_config=model_config,
        strata_col_s=strata_col_cv,
        param_grid=param_grid,
        n_iter=cv_n_iter,
        random_state=random_state,
        cv=cv_n_folds
    )

    print("Random search completed.")

    best_model_wrapper: NextEventSurvivalWrapper = random_search.best_estimator_
    best_model: NextEventPredictionModel = best_model_wrapper.model

    rand_search_results = summarize_search_results(random_search, print_results=True, model_name=model_config["model_name"])

    best_model.random_search_cv_results = rand_search_results

    if hasattr(best_model.model, 'plot') and callable(getattr(best_model.model, 'plot', None)):
        ax = best_model.model.plot()
        print("Model coefficients plot saved.")
        save_coef_lifelines_plot(ax, best_model)
    
    save_pred_output_files_to_csv(best_model, X_test, main_evaluation_time, neptune_run)
    plot_surv_curves_examples(best_model, X_test, duration_col, event_col, neptune_run)

    plot_feature_importance_survival_model(
        best_model,
        model_name,
        X_train,
        plot_shap=plot_shap,
        neptune_run=neptune_run,
        neptune_path='feature_importance',
        evaluation_time=main_evaluation_time,
    )

    evaluate_model(
        best_model,
        X_train,
        X_test,
        evaluation_times=evaluation_times,
        evaluation_set="test",
        save_plots=True,
        show_plot=False,
    )
    evaluate_model_around_specific_time(
        best_model,
        X_test,
        main_evaluation_time,
        save_plots=True,
        evaluation_set="test",
        show_plot=False,
        event_name=event_name,
    )

    model_file_path = best_model.save_model()
    best_model.save_model_params()

    key_test_performance_metrics = best_model.key_test_performance_metrics
    key_train_performance_metrics = best_model.key_train_performance_metrics

    if neptune_run:
        neptune_run["metrics/evaluation"] = key_test_performance_metrics
        neptune_run["metrics/train"] = key_train_performance_metrics
        neptune_run["metrics/cv_results"] = rand_search_results
        upload_model_to_neptune(neptune_run, model_file_path)
        best_model_dir = best_model.get_model_dir()
        upload_evaluation_outputs_to_neptune(neptune_run, best_model_dir, main_evaluation_time)
        upload_params_outputs_to_neptune(neptune_run, best_model_dir)


def main(
    dataset: str,
    model_config_path: str,
    training_data_filepath: str,
    log_in_neptune=True,
    neptune_tags=None,
    random_state=42,
    preprocess_encode_features=True,
    use_fixed_train_test_splits=False,
    split_csv_path=None,
):
    print(f"Starting training for dataset: {dataset}")
    print(f"Log Neptune: {log_in_neptune}")
    print(f"Model config path: {model_config_path}")
    print(f"Training data path: {training_data_filepath}")
    print(f"Random state: {random_state}")

    training_df = pd.read_csv(training_data_filepath)

    if dataset == "mimic":
        if preprocess_encode_features:
            features_to_encode = [
                "GENDER",
                "INSURANCE",
                "ETHNICITY",
                "ADMISSION_TYPE",
                "DISCHARGE_LOCATION",
            ]  # Add any categorical features you want to one-hot encode
            one_hot_cols_to_drop = [
                "GENDER_F",
                "INSURANCE_SELF_PAY",
                "ETHNICITY_OTHER",
                "ADMISSION_TYPE_EMERGENCY",
                "DISCHARGE_LOCATION_OTHERS",
            ]
            training_df = remap_discharge_location(training_df)
            training_df = remap_mimic_races(training_df)
            training_df, _ = preprocess_features_to_one_hot_encode(
                training_df,
                features_to_encode,
                one_hot_cols_to_drop,
            )
        plot_shap = True  # Set to False if you do not want to plot SHAP feature importance
        missing_features = None  # Set to None if no missing features to impute
    elif dataset == "relapse":
        missing_features = {
            "LOG_TIME_RELAPSE_PAST_MEAN": "median",
        }
        plot_shap = True

    model_config = import_yaml_config(model_config_path)
    training_df = remap_strata_col(training_df, model_config)

    neptune_run = None
    if log_in_neptune:
        model_type: str = model_config["model_type"]
        model_name: str = model_config["model_name"]

        run_name = model_name.lower().replace(" ", "_")
        tags=["survival_model"] if neptune_tags is None else neptune_tags
        if model_type not in tags:
            tags.append(model_type)
        data_config_path = "/workspaces/master-thesis-recurrent-health-events-prediction/recurrent_health_events_prediction/configs/data_config.yaml"
        neptune_run = initialize_neptune_run(data_config_path, run_name, dataset, tags=tags) if log_in_neptune else None

        # Log model config and training data path
        neptune_run["training_data/path"] = training_data_filepath

        if neptune_run:
            upload_training_data_to_neptune(
                neptune_run,
                training_data_path=training_data_filepath,
                base_neptune_path="artifacts/traditional_survival_model/training_data",
            )
            # track_file_in_neptune(
            #     neptune_run,
            #     "artifacts/traditional_survival_model/training_data",
            #     training_data_filepath,
            # )

    train_traditional_survival_model(
        training_df=training_df,
        model_config=model_config,
        random_state=random_state,
        cv_n_folds=5,
        cv_n_iter=15,
        q_bins=1,
        plot_shap=plot_shap,
        neptune_run=neptune_run,
        missing_features=missing_features,
        split_csv_path=split_csv_path,
        use_fixed_train_test_split=use_fixed_train_test_splits
    )

    if neptune_run:
        neptune_run.stop()

    print("Training completed successfully.")

if __name__ == "__main__":
    dataset = "mimic"  # Change to "mimic" for MIMIC-III dataset
    neptune_tags = ["hmm_mimic_time_log_gamma", "cox_ph", "fixed_train_test_split", "final_report", "hmm_survival_model"]  # Add any tags you want for the Neptune run
    log_in_neptune = True  # Set to True to log the run in Neptune
    random_state = 64  # Set random state for reproducibility
    preprocess_encode_features = False  # Set to True if you want to one-hot encode categorical features
    use_fixed_train_test_splits = True  # Set to True to use a fixed train-test split from a CSV file

    if dataset == "mimic":
        hmm_survival = True  # Set to True if using with HMM survival model
        model_name = "cox_ph_with_hmm_log_gamma_covs_model" # Specify the model name for MIMIC-III dataset
        model_subdir = "hmm_survival" if hmm_survival else "survival"
        split_csv_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/train_test_split.csv"
        model_config_filepath = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/mimic/{model_subdir}/{model_name}/{model_name}_config.yaml"
        if hmm_survival:
            training_data_filepath = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/mimic/{model_subdir}/{model_name}/last_events_with_hidden_states.csv"
        else:
            # Path to the preprocessed MIMIC-III dataset for COPD and heart failure patients
            training_data_filepath = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/last_events.csv"
    else:
        hmm_survival = True  # Set to True if using with HMM survival model
        model_name = "hmm_binary_30_days_relapse_aft_weibull_model" # Specify the model name for relapse dataset
        model_subdir = "hmm_survival" if hmm_survival else "survival"
        model_config_filepath = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/drug_relapse/{model_subdir}/{model_name}/{model_name}_config.yaml"
        split_csv_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/train_test_split.csv"
        if hmm_survival:
            training_data_filepath = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/drug_relapse/hmm_survival/{model_name}/last_events_with_hidden_states.csv"
        else:
            training_data_filepath = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/last_relapses.csv"

    model_config_exists = check_if_file_exists(model_config_filepath)
    training_data_exists = check_if_file_exists(training_data_filepath)
    if not model_config_exists:
        raise FileNotFoundError(f"Model configuration file not found: {model_config_filepath}")
    if not training_data_exists:
        raise FileNotFoundError(f"Training data directory not found: {training_data_filepath}")

    main(
        dataset=dataset,
        model_config_path=model_config_filepath,
        training_data_filepath=training_data_filepath,
        neptune_tags=neptune_tags,
        random_state=random_state,
        log_in_neptune=log_in_neptune,
        preprocess_encode_features=preprocess_encode_features,
        use_fixed_train_test_splits=use_fixed_train_test_splits,
        split_csv_path=split_csv_path
    ) 
