from typing import Optional
import yaml
import neptune
import pandas as pd
import numpy as np
import os
import warnings
from importlib import resources as impresources

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.model.utils import plot_auc
from recurrent_health_events_prediction.utils.general_utils import check_if_file_exists

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from recurrent_health_events_prediction.training.utils import (
    find_best_threshold,
    plot_calibration_curve,
    plot_pred_proba_distribution,
    summarize_search_results
)
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_plotly_plots_to_neptune_run,
    initialize_neptune_run,
    track_file_in_neptune,
    upload_file_to_neptune,
)

from recurrent_health_events_prediction.training.utils_traditional_classifier import (
    add_cv_and_evaluation_results_to_neptune,
    add_training_data_stats_to_neptune,
    extract_results,
    impute_missing_features,
    plot_all_feature_importances,
    save_test_predictions,
    scale_features,
)

def train_test_classifier(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param_distributions: dict,
    rand_search_config: dict,
    neptune_run: Optional[neptune.Run] = None,
    verbose=False,
    plot_shap=True,
    random_state=42,
    class_names: Optional[list[str]] = None,
    show_plots: bool = False,
):
    print(f"\nTraining {model_name} with hyperparameter search...")
    
    random_search = RandomizedSearchCV(
        model,
        param_distributions,
        random_state=random_state,
        cv=rand_search_config.get("cv", 5),
        scoring=rand_search_config.get("scoring", "roc_auc"),
        n_iter=rand_search_config.get("n_iter", 10),
        n_jobs=rand_search_config.get("n_jobs", 3),
    )
    try:
        random_search.fit(X_train, y_train)
    except ValueError as e:
        print(f"Error during {model_name} training:", e)
        if "Input X contains NaN" in str(e):
            nan_cols = X_train.columns[X_train.isna().any()].tolist()
            print("Columns with NaN values in X_train:", nan_cols)
            raise ValueError(
                f"X_train contains NaN values. Please check your data preprocessing."
            )
    
    cv_search_results = summarize_search_results(
        random_search, print_results=verbose, model_name=model_name
    )
    
    # Inference and evaluation
    y_pred_proba = random_search.predict_proba(X_test)
    if y_pred_proba.ndim != 1:
        y_pred_proba = y_pred_proba[:, 1]  # Get probabilities for the positive class
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"{model_name} model evaluation:")
    print("AUC: ", roc_auc)
    
    # Find best threshold based on F1 score
    best_threshold, best_f1 = find_best_threshold(y_test, y_pred_proba)
    print(f"Best threshold for F1 score: {best_threshold:.4f} with F1: {best_f1:.4f}")
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Compute metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    
    plot_all_feature_importances(
            random_search.best_estimator_,
            X_train,
            y_train,
            model_name=model_name,
            neptune_run=neptune_run,
            neptune_path="feat_importances",
            random_state=rand_search_config.get("random_state", 42),
            plot_shap=plot_shap,
            show_plots=show_plots
        )
    class_names_dict = {i: class_names[i] for i in range(len(class_names))} if class_names else None
    fig_hist = plot_pred_proba_distribution(y_test, y_pred_proba, show_plot=False, class_names=class_names_dict)
    fig_hist = fig_hist.update_layout(title=f"Predicted Probabilities by True Labels - {model_name}")
    fig_auc = plot_auc(y_pred_proba, y_test, show_plot=False, title=f"ROC Curve - {model_name}")
    fig_cal = plot_calibration_curve(y_test, y_pred_proba, show_plot=False, title=f"Calibration Curve - {model_name}")

    if neptune_run:
        # Upload plots to Neptune
        neptune_path = f"results/{model_name.lower().replace(' ', '_')}/evaluation"
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_hist,
            filename="pred_proba_distribution.html",
            filepath=neptune_path
        )
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_auc,
            filename="roc_curve.html",
            filepath=neptune_path
        )
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_cal,
            filename="calibration_curve.html",
            filepath=neptune_path
        )

    eval_results = {
        "roc_auc": roc_auc,
        "f1_score": f1,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }
    
    if neptune_run:
        add_cv_and_evaluation_results_to_neptune(
            neptune_run,
            model_name,
            best_threshold,
            cv_search_results,
            eval_results,
            neptune_path="results",
            conf_matrix=conf_matrix,
            class_names=class_names,
        )

    print(f"Finished training and evaluation of {model_name}.\n")
    return y_pred_proba, y_pred, eval_results, cv_search_results

def train_test_pipeline(
    models_to_train: Optional[list[str]],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_config: dict,
    target_col: str,
    event_id_col: str,
    output_dir: str,
    use_hmm_features=False,
    features_to_scale: Optional[list[str]] = None,
    random_state=42,
    plot_shap=True,
    neptune_run: Optional[neptune.Run] = None,
    missing_features: Optional[dict[str]] = None,
    class_names: Optional[list[str]] = None,
):
    base_features_cols: list[str] = (
        model_config["current_feat_cols"] + model_config["past_summary_feat_cols"]
    )

    print("Training traditional classifier...")
    print("Using base features: ", base_features_cols)
    print("Target column: ", target_col)
    print("Using HMM features: ", use_hmm_features)
    print("Random state: ", random_state)

    # Add a random feature for reference
    np.random.seed(random_state)  # Ensure reproducibility
    train_df["RANDOM_FEATURE"] = np.random.rand(len(train_df))
    test_df["RANDOM_FEATURE"] = np.random.rand(len(test_df))

    if neptune_run:
        neptune_run["base_features"] = "\n".join(base_features_cols)
        neptune_run["random_state"] = random_state
        neptune_run["target_col"] = target_col

    # Check if HMM features are to be used
    hmm_prob_features = []
    if use_hmm_features:
        hmm_prob_features = [
            col for col in train_df.columns if "_HIDDEN_RISK_" in col
        ]
        print("Using HMM probability features: ", hmm_prob_features)
        if neptune_run:
            neptune_run["hmm_prob_features"] = "\n".join(
                hmm_prob_features
            )
        features_to_scale = features_to_scale + hmm_prob_features if features_to_scale else None

    feature_cols = base_features_cols + ["RANDOM_FEATURE"] + hmm_prob_features

    if neptune_run:
        add_training_data_stats_to_neptune(
            neptune_run, train_df, target_col, feature_cols, neptune_path="training_data_stats/train"
        )
        
        add_training_data_stats_to_neptune(
            neptune_run, test_df, target_col, feature_cols, neptune_path="training_data_stats/test"
        )

    # Train set
    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int)
    # Test set
    X_test = test_df[feature_cols]
    y_test = test_df[target_col].astype(int)
    test_ids = test_df[event_id_col]

    if missing_features:
        X_train, X_test = impute_missing_features(X_train, X_test, missing_features)
    
    features_to_scale = features_to_scale + ["RANDOM_FEATURE"] if features_to_scale is not None else None

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, features_to_scale=features_to_scale
    )

    print("Class distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts(normalize=True))
    
    print("\nTraining models...\n")

    y_pred_proba_dict = {}
    y_pred_dict = {}

    # ===== Logistic Regression =====
    if "logistic_regression" in models_to_train:
        additional_params: dict = model_config.get("log_reg", {}).get("additional_params", {})
        logreg_model = LogisticRegression(
            random_state=random_state, **additional_params
        )

        param_distributions = model_config["logistic_regression"]["param_distributions"]
        param_distributions["C"] = np.logspace(-4, 4, 100)  # 100 values between 1e-4 and 1e4
        rand_search_config = model_config["logistic_regression"]["rand_search_config"]

        y_pred_proba_logreg, y_pred_logreg, eval_results_logreg, cv_search_results_logreg = train_test_classifier(
            logreg_model,
            "Logistic Regression",
            X_train_scaled,
            y_train,
            X_test_scaled,
            y_test,
            param_distributions,
            rand_search_config,
            neptune_run=neptune_run,
            verbose=True,
            plot_shap=plot_shap,
            random_state=random_state,
            class_names=class_names
        )
        y_pred_proba_dict["logreg"] = y_pred_proba_logreg
        y_pred_dict["logreg"] = y_pred_logreg

    # ===== Random Forest =====
    if "random_forest" in models_to_train:
        additional_params: dict = model_config.get("random_forest", {}).get("additional_params", {})
        rf_model = RandomForestClassifier(random_state=42, **additional_params)

        param_distributions = model_config["random_forest"]["param_distributions"]
        rand_search_config = model_config["random_forest"]["rand_search_config"]

        y_pred_proba_rf, y_pred_rf, eval_results_rf, cv_search_results_rf = train_test_classifier(
            rf_model,
            "Random Forest",
            X_train,
            y_train,
            X_test,
            y_test,
            param_distributions,
            rand_search_config,
            neptune_run=neptune_run,
            verbose=True,
            plot_shap=plot_shap,
            random_state=random_state,
            class_names=class_names
        )
        y_pred_proba_dict["rf"] = y_pred_proba_rf
        y_pred_dict["rf"] = y_pred_rf

    # ===== LightGBM =====
    if "lgbm" in models_to_train:
        additional_params: dict = model_config.get("lgbm", {}).get("additional_params", {})
        lgbm = LGBMClassifier(
            random_state=random_state,
            **additional_params
        )

        param_distributions = model_config["lgbm"]["param_distributions"]
        rand_search_config = model_config["lgbm"]["rand_search_config"]

        y_pred_proba_lgbm, y_pred_lgbm, eval_results_lgbm, cv_search_results_lgbm = train_test_classifier(
            lgbm,
            "LightGBM",
            X_train,
            y_train,
            X_test,
            y_test,
            param_distributions,
            rand_search_config,
            neptune_run=neptune_run,
            verbose=True,
            plot_shap=plot_shap,
            random_state=random_state,
            class_names=class_names
        )
        
        y_pred_proba_dict["lgbm"] = y_pred_proba_lgbm
        y_pred_dict["lgbm"] = y_pred_lgbm

    # ===== Save All Predictions =====
    pred_test_output_filepath = os.path.join(output_dir, "test_predictions.csv")
    print("\nSaving test predictions...")
    print("Output path for test predictions: ", pred_test_output_filepath)
    save_test_predictions(
        out_path=pred_test_output_filepath,
        id_series=test_ids,
        y_true=y_test,
        proba_dict=y_pred_proba_dict,
        pred_dict=y_pred_dict,
        file_format="csv"  # Change to "parquet" if needed
    )
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            pred_test_output_filepath,
            neptune_base_path="artifacts/inference",
            neptune_filename="test_predictions.csv"
        )
    
    # ===== Summarize Results =====
    results_df = pd.DataFrame()
    if "logistic_regression" in models_to_train:
        dict_logreg_metrics = extract_results(eval_results_logreg, cv_search_results_logreg)
        results_df = pd.concat([results_df, pd.DataFrame([dict_logreg_metrics])], ignore_index=True)

    if "random_forest" in models_to_train:
        dict_rf_metrics = extract_results(eval_results_rf, cv_search_results_rf)
        results_df = pd.concat([results_df, pd.DataFrame([dict_rf_metrics])], ignore_index=True)

    if "lgbm" in models_to_train:
        dict_lgbm_metrics = extract_results(eval_results_lgbm, cv_search_results_lgbm)
        results_df = pd.concat([results_df, pd.DataFrame([dict_lgbm_metrics])], ignore_index=True)

    results_output_filepath = os.path.join(output_dir, "performance_results.csv")
    print("\nSaving performance results...")
    print("Output path for performance results: ", results_output_filepath)
    results_df.to_csv(results_output_filepath, index=False)

def main(
    models_to_train: list[str],
    dataset: str,
    model_config_path: Optional[str],
    output_dir: str,
    log_in_neptune=True,
    neptune_tags=[],
    random_state=42,
    use_hmm_features=False,  # Set to True if you want to use HMM features
    plot_shap=True,  # Set to False if you do not want to plot SHAP feature importance
):
    data_config_path = (impresources.files(configs) / "data_config.yaml")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    training_data_config = data_config['training_data'][dataset]
    train_test_data_dir = training_data_config["data_directory"]
    
    target_col = training_data_config["binary_event_col"]  # Variable to predict
    event_id_col = training_data_config["hosp_id_col"]
    class_names= training_data_config.get("class_names", None)
    features_to_scale = training_data_config.get('features_to_scale', None)
    missing_features = model_config.get('missing_features', None)
    next_admt_type_col = training_data_config.get("next_admt_type_col", None)
    
    train_data_filepath = os.path.join(train_test_data_dir, "train_events.csv")
    test_data_filepath = os.path.join(train_test_data_dir, "test_events.csv")
    
    print("Dataset: ", dataset)
    print("Loading training data from: ", train_data_filepath)
    print("Loading test data from: ", test_data_filepath)
    
    train_df = pd.read_csv(train_data_filepath)
    train_df = train_df[(train_df["IS_LAST_EVENT"] == 1) & (train_df[next_admt_type_col] != "ELECTIVE")]
    test_df = pd.read_csv(test_data_filepath)
    test_df = test_df[(test_df["IS_LAST_EVENT"] == 1) & (test_df[next_admt_type_col] != "ELECTIVE")]

    print("Output directory: ", output_dir)

    tags = (
        ["baseline", "traditional_classifiers"]
        if neptune_tags is None
        else neptune_tags
    )
    run_name = "baseline_classifiers"
    neptune_run = (
        initialize_neptune_run(data_config_path, run_name, dataset, tags=tags)
        if log_in_neptune
        else None
    )
    if log_in_neptune:
        # Log model config and training data path
        neptune_run["data_directory"] = train_test_data_dir
        neptune_run["model_config"] = model_config
        upload_file_to_neptune(
            neptune_run,
            model_config_path,
            neptune_base_path="artifacts/configs",
            neptune_filename="model_config.yaml",
        )

        track_file_in_neptune(
            neptune_run,
            "artifacts/data/train",
            train_data_filepath,
        )
        track_file_in_neptune(
            neptune_run,
            "artifacts/data/test",
            test_data_filepath,
        )

    train_test_pipeline(
        train_df=train_df,
        test_df=test_df,
        model_config=model_config,
        output_dir=output_dir,
        target_col=target_col,
        event_id_col=event_id_col,
        features_to_scale=features_to_scale,
        use_hmm_features=use_hmm_features,
        random_state=random_state,
        plot_shap=plot_shap,
        neptune_run=neptune_run,  # Replace with your Neptune run if applicable
        missing_features=missing_features,
        class_names=class_names,
        models_to_train=models_to_train,
    )

    if neptune_run:
        print("Stopping Neptune run...")
        neptune_run.stop()


if __name__ == "__main__":
    dataset = "mimic"  # Change to "mimic" if you want to run on MIMIC dataset or "relapse" for relapse dataset
    multiple_hosp_patients = True  # True if patients can have multiple hospital admissions
    # Define Neptune tags for logging
    neptune_tags = [
        "baseline",
        dataset,
        "last_events",
        "traditional_classifiers",
        "test_run"
    ]
    random_state = 42  # Set random state for reproducibility
    log_in_neptune = True  # Set to True to log the run in Neptune

    if multiple_hosp_patients:
        neptune_tags.append("multiple_hosp_patients")
    else:
        neptune_tags.append("all_patients")

    if dataset == "mimic":
        model_config_path = "/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/classifier_baselines/config.yaml"
    if dataset == "relapse":
        model_config_path = "/workspaces/msc-thesis-recurrent-health-modeling/_models/relapse/classifier_baselines/config.yaml"

    model_config_exists = check_if_file_exists(model_config_path)
    if not model_config_exists:
        raise FileNotFoundError(f"Model config file not found: {model_config_path}")
    
    output_dir = os.path.dirname(model_config_path)
    if multiple_hosp_patients:
        output_dir += "/multiple_hosp_patients"
        os.makedirs(output_dir, exist_ok=True)

    models_to_train = [
        "logistic_regression",
        "random_forest",
        # "lgbm",
    ]

    main(
        dataset=dataset,
        model_config_path=model_config_path,
        output_dir=output_dir,
        log_in_neptune=log_in_neptune,
        neptune_tags=neptune_tags,
        random_state=random_state,
        use_hmm_features = False,  # Set to True if you want to use HMM features
        models_to_train=models_to_train,
    )
