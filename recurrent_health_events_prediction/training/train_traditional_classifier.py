from typing import Optional
import neptune
import pandas as pd
import numpy as np
import os
import warnings

from recurrent_health_events_prediction.preprocessing.utils import remap_discharge_location, remap_mimic_races
from recurrent_health_events_prediction.utils.general_utils import check_if_file_exists

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

from recurrent_health_events_prediction.training.utils import (
    apply_train_test_split_file_classification,
    preprocess_features_to_one_hot_encode,
    summarize_search_results
)
from recurrent_health_events_prediction.utils.neptune_utils import (
    initialize_neptune_run,
    track_file_in_neptune,
    upload_file_to_neptune,
    upload_training_data_to_neptune,
)

from recurrent_health_events_prediction.training.utils_traditional_classifier import (
    add_cv_and_evaluation_results_to_neptune,
    add_training_data_stats_to_neptune,
    impute_missing_features,
    plot_all_feature_importances,
    save_prob_predictions,
    scale_features,
)

def train_traditional_classifier(
    training_df: pd.DataFrame,
    base_features_cols: list[str],
    target_col: str,
    event_id_col: str,
    use_hmm_features=False,
    random_state=42,
    cv_scoring="roc_auc",
    cv_n_folds=5,
    cv_n_iter=25,
    test_size=0.2,
    plot_shap=True,
    features_not_to_scale: Optional[list[str]] = None,
    neptune_run: Optional[neptune.Run] = None,
    neptune_path="traditional_classifier",
    missing_features: Optional[dict[str]] = None,
    use_fixed_train_test_split: bool = False,
    split_csv_path: Optional[str] = None,
    output_prob_predictions_path: Optional[str] = None,
):

    print("Training traditional classifier...")
    print("Using base features: ", base_features_cols)
    print("Target column: ", target_col)
    print("Using HMM features: ", use_hmm_features)
    print("Random state: ", random_state)

    # Add a random feature for reference
    np.random.seed(random_state)  # Ensure reproducibility
    training_df["RANDOM_FEATURE"] = np.random.rand(len(training_df))

    if neptune_run:
        neptune_run[f"{neptune_path}/base_features"] = "\n".join(base_features_cols)
        neptune_run[f"{neptune_path}/random_state"] = random_state
        neptune_run[f"{neptune_path}/target_col"] = target_col

    # Check if HMM features are to be used
    hmm_prob_features = []
    if use_hmm_features:
        hmm_prob_features = [
            col for col in training_df.columns if "_HIDDEN_RISK_" in col
        ]
        print("Using HMM probability features: ", hmm_prob_features)
        if neptune_run:
            neptune_run[f"{neptune_path}/hmm_prob_features"] = "\n".join(
                hmm_prob_features
            )

    feature_cols = base_features_cols + ["RANDOM_FEATURE"] + hmm_prob_features

    if neptune_run:
        add_training_data_stats_to_neptune(
            neptune_run, training_df, target_col, feature_cols
        )

    if use_fixed_train_test_split:
        if split_csv_path is None:
            raise ValueError("split_csv_path must be provided when use_fixed_train_test_split is True")
        print(f"Using fixed train-test split from {split_csv_path}")
        X_train, X_test, y_train, y_test, _, test_ids = (
            apply_train_test_split_file_classification(
                training_df, split_csv_path, event_id_col, target_col, feature_cols
            )
        )
    else:
        print("Using random train-test split...")
        X = training_df[feature_cols + [event_id_col]]
        y = training_df[target_col].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        test_ids = X_test[event_id_col].values
        X_train = X_train.drop(columns=[event_id_col])
        X_test = X_test.drop(columns=[event_id_col])

    if missing_features:
        X_train, X_test = impute_missing_features(X_train, X_test, missing_features)

    X_train_scaled, X_test_scaled = scale_features(
        X_train, X_test, features_not_to_scale=features_not_to_scale
    )

    print("Class distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("\nClass distribution in test set:")
    print(pd.Series(y_test).value_counts(normalize=True))

    # ===== Logistic Regression =====
    print("Training Logistic Regression with hyperparameter search...")
    logreg_model = LogisticRegression(
        random_state=random_state, class_weight="balanced", max_iter=1000
    )

    rand_search_params = {
        "C": np.logspace(-4, 4, 100),  # 100 values between 1e-4 and 1e4
        "penalty": ["l2", "l1"],  # 'l1' if solver supports it
        "solver": ["liblinear"],  # 'saga' supports 'l1' penalty
        "fit_intercept": [True, False],
    }

    random_search_logreg = RandomizedSearchCV(
        logreg_model,
        rand_search_params,
        cv=cv_n_folds,
        scoring=cv_scoring,
        n_iter=cv_n_iter,
        random_state=random_state,
        n_jobs=5,
    )

    try:
        random_search_logreg.fit(X_train_scaled, y_train)
    except ValueError as e:
        print("Error during Logistic Regression training:", e)
        if "Input X contains NaN" in str(e):
            nan_cols = X_train_scaled.columns[X_train_scaled.isna().any()].tolist()
            print("Columns with NaN values in X_train_scaled:", nan_cols)
            raise ValueError(
                "X_train_scaled contains NaN values. Please check your data preprocessing."
            )

    logreg_cv_search_results = summarize_search_results(
        random_search_logreg, print_results=False, model_name="Logistic Regression"
    )
    # Inference and evaluation Logistic Regression
    y_pred_proba_logreg = random_search_logreg.predict_proba(X_test_scaled)
    if y_pred_proba_logreg.ndim != 1:
        y_pred_proba_logreg = y_pred_proba_logreg[:, 1]  # Get probabilities for the positive class

    roc_auc_logreg = roc_auc_score(y_test, y_pred_proba_logreg)
    print("Logistic Regression model evaluation:")
    print("AUC: ", roc_auc_logreg)
    if neptune_run:
        plot_all_feature_importances(
            random_search_logreg.best_estimator_,
            X_train_scaled,
            y_train,
            model_name="Logistic Regression",
            neptune_run=neptune_run,
            random_state=random_state,
            plot_shap=plot_shap,
        )
        add_cv_and_evaluation_results_to_neptune(
            neptune_run,
            "Logistic Regression",
            logreg_cv_search_results,
            roc_auc_logreg,
            neptune_path=neptune_path,
        )

    # ===== Random Forest =====
    print("Training Random Forest with hyperparameter search...")
    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")

    rand_search_params = {
        "n_estimators": [50, 100, 200],
        #'n_estimators': [100, 200, 250, 300],
        "max_depth": [None, 3, 5, 10, 12],
        "min_samples_split": [2, 5, 10, 15, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    random_search_rf = RandomizedSearchCV(
        rf_model,
        rand_search_params,
        cv=cv_n_folds,
        scoring=cv_scoring,
        n_iter=cv_n_iter,
        random_state=random_state,
        n_jobs=5,
    )
    random_search_rf.fit(X_train, y_train)
    rf_cv_search_results = summarize_search_results(
        random_search_rf, print_results=False, model_name="Random Forest"
    )

    # Inference and evaluation Random Forest
    y_pred_proba_rf = random_search_rf.predict_proba(
        X_test
    )  # shape: (n_samples, n_classes)
    if y_pred_proba_rf.ndim != 1:
        y_pred_proba_rf = y_pred_proba_rf[:, 1]  # Get probabilities for the positive class
    roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    print("Random Forest model evaluation:")
    print("AUC: ", roc_auc_rf)

    if neptune_run:
        plot_all_feature_importances(
            random_search_rf.best_estimator_,
            X_train,
            y_train,
            model_name="Random Forest",
            neptune_run=neptune_run,
            random_state=random_state,
            plot_shap=plot_shap,
        )
        add_cv_and_evaluation_results_to_neptune(
            neptune_run,
            "Random Forest",
            rf_cv_search_results,
            roc_auc_rf,
            neptune_path=neptune_path,
        )

    # ===== LightGBM =====
    print("Training LGBM Classifier with hyperparameter search...")
    lgbm_param_grid = {
        "n_estimators": [50, 100, 200],
        #'n_estimators': [100, 200, 250, 300],
        "max_depth": [3, 4, 6, -1],
        "min_child_samples": [5, 10, 20, 30],
        "num_leaves": [7, 15, 31],
        "subsample": [0.6, 0.8, 1.0],
        "feature_fraction": [0.6, 0.8, 1.0],
    }

    lgbm = LGBMClassifier(
        random_state=random_state,
        class_weight="balanced",
        importance_type="gain",
        verbose=-1,
    )

    random_search_lgbm = RandomizedSearchCV(
        lgbm,
        lgbm_param_grid,
        n_iter=5,  # Increase for more thorough search
        scoring=cv_scoring,
        cv=cv_n_folds,
        random_state=42,
        n_jobs=5,  # Use all cores
    )

    # Train the model
    random_search_lgbm.fit(X_train, y_train)
    lgbm_cv_search_results = summarize_search_results(
        random_search_lgbm, print_results=False, model_name="LightGBM"
    )
    # Inference and evaluation LGBM
    y_pred_proba_lgbm = random_search_lgbm.predict_proba(
        X_test
    )  # shape: (n_samples, n_classes)
    if y_pred_proba_lgbm.ndim != 1:
        y_pred_proba_lgbm = y_pred_proba_lgbm[:, 1]
    roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
    print("LightGBM model evaluation:")
    print("AUC: ", roc_auc_lgbm)

    if neptune_run:
        plot_all_feature_importances(
            random_search_lgbm.best_estimator_,
            X_train,
            y_train,
            model_name="LightGBM",
            neptune_run=neptune_run,
            random_state=random_state,
            plot_shap=plot_shap,
        )
        add_cv_and_evaluation_results_to_neptune(
            neptune_run,
            "LightGBM",
            lgbm_cv_search_results,
            roc_auc_lgbm,
            neptune_path=neptune_path,
        )
    if use_fixed_train_test_split:
        print("Saving probability predictions...")
        print("Output path for probability predictions: ", output_prob_predictions_path)
        if output_prob_predictions_path is None:
            raise ValueError("output_prob_predictions_path must be provided when use_fixed_train_test_split is True")
        y_pred_proba_dict = {
            "logreg": y_pred_proba_logreg,
            "rf": y_pred_proba_rf,
            "lgbm": y_pred_proba_lgbm,
        }
        save_prob_predictions(
            out_path=output_prob_predictions_path,
            id_series=test_ids,
            y_true=y_test,
            proba_dict=y_pred_proba_dict,
            file_format="csv"  # Change to "parquet" if needed
        )
        if neptune_run:
            upload_file_to_neptune(
                neptune_run,
                output_prob_predictions_path,
                neptune_base_path="artifacts/traditional_classifier/prob_predictions",
                neptune_filename="prob_predictions.csv"
            )


def main(
    dataset: str,
    training_data_filepath: str,
    use_fixed_train_test_split=False,
    log_in_neptune=True,
    neptune_tags=[],
    random_state=42,
):
    missing_features = None  # initialize missing features to None, it will be set later based on the dataset
    split_csv_filepath = (
        os.path.dirname(training_data_filepath) + "/train_test_split.csv"
    )

    if dataset == "mimic":
        training_df = pd.read_csv(training_data_filepath)
        base_features_cols = [
            "AGE",
            "CHARLSON_INDEX",
            "NUM_PREV_HOSPITALIZATIONS",
            "LOG_DAYS_SINCE_LAST_HOSPITALIZATION",
            "NUM_PROCEDURES",
            "LOG_DAYS_IN_ICU",
            "NUM_DRUGS",
            "HAS_DIABETES",
            "HAS_COPD",
            "HAS_CONGESTIVE_HF",
            "READM_30_DAYS_PAST_MEAN",
            "READM_30_DAYS_PAST_SUM",
            "PREV_READMISSION_30_DAYS",
            "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_MEDIAN",
            "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_STD",
        ]
        target_col = (
            "READMISSION_30_DAYS"  # Variable to predict
        )
        event_id_col = "HADM_ID"
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
        training_df, new_cols = preprocess_features_to_one_hot_encode(
            training_df,
            features_to_encode,
            one_hot_cols_to_drop,
        )
        features_not_to_scale = [
            "HAS_DIABETES",
            "HAS_COPD",
            "HAS_CONGESTIVE_HF",
            "PREV_READMISSION_30_DAYS",
        ] + new_cols
        base_features_cols += new_cols
        plot_shap = (
            True  # Set to False if you do not want to plot SHAP feature importance
        )
        use_hmm_features = False  # Set to True if you want to use HMM features
        missing_features = {
            "READM_30_DAYS_PAST_MEAN": "median",
            "READM_30_DAYS_PAST_SUM": "median",
            "LOG_DAYS_SINCE_LAST_HOSPITALIZATION": "median",
            "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_MEDIAN": "median",
            "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_STD": "median",
        }

    elif dataset == "relapse":
        training_df = pd.read_csv(training_data_filepath)
        base_features_cols = [
            "NUM_PREV_RELAPSES",
            "PREV_NUM_DRUGS_POSITIVE",
            "DRUG_POSITIVE_PAST_MEAN",
            "DRUG_POSITIVE_PAST_SUM",
            "NUM_POSITIVES_SINCE_LAST_NEGATIVE",
            "LOG_PARTICIPATION_DAYS",
            "LOG_TIME_SINCE_LAST_NEGATIVE",
            "LOG_TIME_SINCE_LAST_POSITIVE",
            "AGE",
            "LOG_TIME_RELAPSE_PAST_MEAN",
            "LOG_TIME_RELAPSE_PAST_MEDIAN",
            "LOG_TIME_RELAPSE_PAST_STD",
            "PREV_RELAPSE_30_DAYS",
            "RELAPSE_30_DAYS_PAST_MEAN",
            "RELAPSE_30_DAYS_PAST_SUM"
        ]
        # features_to_encode = ['PROGRAM_TYPE', 'RACE', 'GENDER']
        target_col = "RELAPSE_30_DAYS"  # Variable to predict
        event_id_col = "COLLECTION_ID"
        use_hmm_features = False  # Set to True if you want to use HMM features
        # training_df, new_cols = one_hot_encode_feature(training_df, features_to_encode)
        # features_not_to_scale = new_cols
        features_not_to_scale = ["PREV_RELAPSE_30_DAYS"]
        # base_features_cols += new_cols
        plot_shap = True
        missing_features = {
            "LOG_TIME_RELAPSE_PAST_STD": "median",
            "RELAPSE_30_DAYS_PAST_MEAN": "median",
            "LOG_TIME_RELAPSE_PAST_MEAN": "median",
            "LOG_TIME_RELAPSE_PAST_MEDIAN": "median",
            "RELAPSE_30_DAYS_PAST_SUM": "median",
        }

    data_config_path = "/workspaces/master-thesis-recurrent-health-events-prediction/recurrent_health_events_prediction/configs/data_config.yaml"

    dataset_subdir = "drug_relapse" if dataset == "relapse" else "mimic"
    prob_output_dir = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/{dataset_subdir}/classifiers_baselines"
    prob_output_filepath = os.path.join(prob_output_dir, "prob_predictions.csv")

    tags = (
        ["baseline", "traditional_classifiers"]
        if neptune_tags is None
        else neptune_tags
    )
    run_name = "traditional_classifiers"
    data_config_path = "/workspaces/master-thesis-recurrent-health-events-prediction/recurrent_health_events_prediction/configs/data_config.yaml"
    neptune_run = (
        initialize_neptune_run(data_config_path, run_name, dataset, tags=tags)
        if log_in_neptune
        else None
    )
    if log_in_neptune:
        # Log model config and training data path
        neptune_run["traditional_classifier/training_data/path"] = (
            training_data_filepath
        )

        if neptune_run:
            upload_training_data_to_neptune(
                neptune_run,
                training_data_path=training_data_filepath,
                base_neptune_path="artifacts/traditional_classifier/training_data",
            )
            track_file_in_neptune(
                neptune_run,
                "artifacts/traditional_classifier/training_data/training_file",
                training_data_filepath,
            )

    train_traditional_classifier(
        training_df=training_df,
        base_features_cols=base_features_cols,
        target_col=target_col,
        event_id_col=event_id_col,
        use_hmm_features=use_hmm_features,
        random_state=random_state,
        cv_scoring="roc_auc",
        cv_n_folds=5,
        cv_n_iter=15,
        test_size=0.2,
        features_not_to_scale=features_not_to_scale,
        plot_shap=plot_shap,
        neptune_run=neptune_run,  # Replace with your Neptune run if applicable
        neptune_path="traditional_classifier",
        missing_features=missing_features,
        use_fixed_train_test_split=use_fixed_train_test_split,
        split_csv_path=split_csv_filepath,
        output_prob_predictions_path=prob_output_filepath,
    )

    if neptune_run:
        print("Stopping Neptune run...")
        neptune_run.stop()


if __name__ == "__main__":
    dataset = "mimic"  # Change to "mimic" if you want to run on MIMIC dataset or "relapse" for relapse dataset
    # Define Neptune tags for logging
    neptune_tags = [
        "baseline",
        dataset,
        "multiple_hosp_patients",
        "last_events",
    ]
    random_state = 42  # Set random state for reproducibility
    log_in_neptune = True  # Set to True to log the run in Neptune
    use_fixed_train_test_split = True # Set to True to use fixed train-test split
    if dataset == "mimic":
        #training_data_filepath = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/mimic_cleaned/last_events.csv"
        training_data_filepath = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/last_events.csv"
    if dataset == "relapse":
        training_data_filepath = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/last_relapses.csv"
    
    training_data_exists = check_if_file_exists(training_data_filepath)
    if not training_data_exists:
        raise FileNotFoundError(f"Training data directory not found: {training_data_filepath}")
    
    if use_fixed_train_test_split:
        neptune_tags.append("fixed_train_test_split")
    
    main(
        dataset=dataset,
        training_data_filepath=training_data_filepath,
        use_fixed_train_test_split=use_fixed_train_test_split,
        log_in_neptune=log_in_neptune,
        neptune_tags=neptune_tags,
        random_state=random_state,
    )
