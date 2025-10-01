import os
import pandas as pd

from recurrent_health_events_prediction.training.utils import preprocess_features_to_one_hot_encode
from recurrent_health_events_prediction.preprocessing.utils import remap_discharge_location, remap_mimic_races
from recurrent_health_events_prediction.utils.neptune_utils import initialize_neptune_run, track_file_in_neptune
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config, check_if_file_exists, check_if_directory_exists, stringify_dict_values
from utils_hmm import (
    get_training_sequences_stats,
    load_and_prepare_historical_data_mimic,
    load_and_prepare_historical_data_relapse,
    train_and_evaluate_hmm
)

from recurrent_health_events_prediction.preprocessing.gen_dataset_hmm import (
    generate_dataset_hmm_feat,
    load_data_for_inference_mimic,
    load_data_for_inference_relapse,
)
from recurrent_health_events_prediction.training.train_traditional_classifier import (
    train_traditional_classifier,
)


def main(
    dataset,
    model_config_path,
    training_data_path,
    inference_data_path,
    log_in_neptune=False,
    skip_survival_dataset_hmm_generation=False,
    use_fixed_train_test_split=False,
    random_state=35,
    neptune_tags=None,
):
    print("Starting HMM training...")
    print(f"Dataset: {dataset}")
    print(f"Log in Neptune: {log_in_neptune}")
    print(f"Skip survival dataset with HMM features generation: {skip_survival_dataset_hmm_generation}")
    print(f"Model configuration path: {model_config_path}")
    print(f"Training data path: {training_data_path}")
    print(f"Inference data path: {inference_data_path}")

    # Load HMM configuration
    hmm_config = import_yaml_config(model_config_path)

    use_only_sequences_gte_2_steps = hmm_config.get("use_only_sequences_gte_2_steps", False)
    model_name = hmm_config.get("model_name", "hmm_model")

    print("Model name:", model_name)

    tags = (
        ["hmm", "hmm_feat_trad_classifier"]
        if neptune_tags is None
        else neptune_tags
    )
    run_name = model_name
    data_config_path = "/workspaces/master-thesis-recurrent-health-events-prediction/recurrent_health_events_prediction/configs/data_config.yaml"
    neptune_run = (
        initialize_neptune_run(data_config_path, run_name, dataset, tags=tags)
        if log_in_neptune
        else None
    )

    if neptune_run:
        # Log model config and training data path
        neptune_run["model_config"] = hmm_config
        neptune_run["random_state"] = random_state
        neptune_run["training_data/path"] = training_data_path
        neptune_run["emission_variables"] = stringify_dict_values(hmm_config["features"])

    # Load data
    if dataset == "mimic":
        X, filepath_historical_events = load_and_prepare_historical_data_mimic(
            training_data_path,
            use_only_sequences_gt_2_steps=use_only_sequences_gte_2_steps,
        )
    elif dataset == "synthetic":
        filepath_historical_events = os.path.join(training_data_path, "synthetic_test.csv")
        X = pd.read_csv(filepath_historical_events)
    elif dataset == "relapse":
        X, filepath_historical_events = load_and_prepare_historical_data_relapse(
            training_data_path, use_only_sequences_gte_2_steps
        )

    if neptune_run:
        # Log training data statistics
        training_data_stats = get_training_sequences_stats(
            X,
            event_id_col=hmm_config["event_id_col"],
            subject_id_col=hmm_config["id_col"],
        )
        neptune_run["training_data/stats"] = training_data_stats
        neptune_tracking_path = "training_data/files/historical_relapses"
        track_file_in_neptune(neptune_run, neptune_tracking_path, filepath_historical_events)

    initialize_from_first_obs_with_gmm = not use_only_sequences_gte_2_steps
    # Train and evaluate model
    results, hmm = train_and_evaluate_hmm(
        hmm_config=hmm_config,
        X=X,
        neptune_run=neptune_run,
        random_state=random_state,
        initialize_from_first_obs_with_gmm=initialize_from_first_obs_with_gmm
    )
    print("Training completed. Model saved at:", results["model_save_dir"])
    print("Metrics:", results["metrics"])

    if not skip_survival_dataset_hmm_generation:
        missing_features = None  # No missing features to impute in this dataset
        if dataset == "mimic":
            all_events_df, last_events_df, filepath_all_events, filepath_last_events = (
                load_data_for_inference_mimic(inference_data_path)
            )
        elif dataset == "relapse":
            all_events_df, last_events_df, filepath_all_events, filepath_last_events = (
                load_data_for_inference_relapse(inference_data_path)
            )

        if neptune_run:
            track_file_in_neptune(
                neptune_run,
                "artifacts/inference_data/files/all_events",
                filepath_all_events,
            )
            track_file_in_neptune(
                neptune_run,
                "artifacts/inference_data/files/last_events",
                filepath_last_events,
            )

        last_events_with_hmm_feat_df = generate_dataset_hmm_feat(
            all_events_df=all_events_df,
            last_events_df=last_events_df,
            hmm_model=hmm,
            neptune_run=neptune_run,
        )

        if dataset == "mimic":
            base_features_cols = [
                "AGE",
                #"CHARLSON_INDEX",
               # "NUM_PREV_HOSPITALIZATIONS",
                #"LOG_DAYS_SINCE_LAST_HOSPITALIZATION",
                "NUM_PROCEDURES",
                #"LOG_DAYS_IN_ICU",
               # "NUM_DRUGS",
                "HAS_DIABETES",
                "HAS_COPD",
                "HAS_CONGESTIVE_HF",
               # "READM_30_DAYS_PAST_MEAN",
               # "READM_30_DAYS_PAST_SUM",
               # "PREV_READMISSION_30_DAYS",
               # "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_MEDIAN",
               # "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_STD"
            ]
            target_col = (
                "READMISSION_30_DAYS"  # Replace with your actual target column name
            )
            features_to_encode = [
                "GENDER",
                "INSURANCE",
                "ETHNICITY",
                "ADMISSION_TYPE",
                #"DISCHARGE_LOCATION",
            ]  # Add any categorical features you want to one-hot encode
            one_hot_cols_to_drop = [
                "GENDER_F",
                "INSURANCE_SELF_PAY",
                "ETHNICITY_OTHER",
                "ADMISSION_TYPE_EMERGENCY",
                #"DISCHARGE_LOCATION_OTHERS",
            ]

            #last_events_with_hmm_feat_df = remap_discharge_location(
            #    last_events_with_hmm_feat_df
            #)
            last_events_with_hmm_feat_df = remap_mimic_races(
                last_events_with_hmm_feat_df
            )
            last_events_with_hmm_feat_df, new_cols = (
                preprocess_features_to_one_hot_encode(
                    last_events_with_hmm_feat_df,
                    features_to_encode,
                    one_hot_cols_to_drop,
                )
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
            missing_features = {
                "READM_30_DAYS_PAST_MEAN": "median",
                "READM_30_DAYS_PAST_SUM": "median",
                "LOG_DAYS_SINCE_LAST_HOSPITALIZATION": "median",
                "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_MEDIAN": "median",
                "LOG_DAYS_UNTIL_NEXT_HOSP_PAST_STD": "median",
            }
            # missing_features = None  # No missing features to impute in this dataset
        elif dataset == "relapse":
            target_col = 'RELAPSE_30_DAYS'
            base_features_cols = [
                #"NUM_PREV_RELAPSES",
                "PREV_NUM_DRUGS_POSITIVE",
                "DRUG_POSITIVE_PAST_SUM",
                #"NUM_POSITIVES_SINCE_LAST_NEGATIVE",
                "LOG_PARTICIPATION_DAYS",
                #"LOG_TIME_SINCE_LAST_NEGATIVE",
                #"LOG_TIME_SINCE_LAST_POSITIVE",
                "AGE",
                #"LOG_TIME_RELAPSE_PAST_MEAN",
                #"LOG_TIME_RELAPSE_PAST_STD",
                #"PREV_RELAPSE_30_DAYS",
                #"RELAPSE_30_DAYS_PAST_SUM",
                "ALCOHOL_POS",
                "BUPRENORPHINE_POS"
            ]
            features_not_to_scale = ['PREV_RELAPSE_30_DAYS']
            #features_not_to_scale = []
            plot_shap = True
            missing_features = {
                "LOG_TIME_RELAPSE_PAST_STD": "median",
                "LOG_TIME_RELAPSE_PAST_MEAN": "median",
                "RELAPSE_30_DAYS_PAST_SUM": "median"
                }
            
        split_csv_filepath = os.path.join(
            os.path.dirname(training_data_path), "train_test_split.csv"
        )
        prob_output_dir = os.path.dirname(model_config_path)
        prob_output_filepath = os.path.join(prob_output_dir, "prob_predictions.csv")

        train_traditional_classifier(
            last_events_with_hmm_feat_df,
            base_features_cols=base_features_cols,
            target_col = target_col,
            event_id_col = hmm_config["event_id_col"],
            use_hmm_features=True,
            cv_n_iter=10,
            cv_n_folds=5,
            random_state=random_state,
            test_size=0.2,
            cv_scoring='roc_auc',
            plot_shap=plot_shap,
            neptune_run=neptune_run,
            features_not_to_scale = features_not_to_scale,
            missing_features=missing_features,
            use_fixed_train_test_split=use_fixed_train_test_split,
            split_csv_path=split_csv_filepath,
            output_prob_predictions_path=prob_output_filepath,
        )

    if neptune_run:
        # Stop the Neptune run
        neptune_run.stop()


if __name__ == "__main__":
    skip_survival_dataset_hmm_generation = False  # Set to True to skip survival dataset generation and the classifier evaluation
    use_fixed_train_test_split = True  # Set to True to use fixed train-test split
    dataset = "relapse"  # "mimic" # "synthetic" # "relapse"
    log_in_neptune = True  # Set to True to log the run in Neptune
    random_state = 42 # Set random state for reproducibility
    neptune_tags = [
        "hmm",
        "hmm_feat_trad_classifier",
        "multiple_hosp_patients",
        #"multiple_relapses_patients",
        "feature_import_compare",
        #"no_past_summary_features",
        "reduced_feature_set",
        "final_report"
    ]
    if dataset == "mimic":
        model_name = "hmm_mimic_reduced_feats_set"
        model_config_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/mimic/hmm/{model_name}/{model_name}_config.yaml"
        training_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/"
        inference_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/mimic-iii-preprocessed/copd_heart_failure/multiple_hosp_patients/"
    else:
        model_name = "hmm_relapse_reduced_feats_set"
        model_config_path = f"/workspaces/master-thesis-recurrent-health-events-prediction/_models/drug_relapse/hmm/{model_name}/{model_name}_config.yaml"
        training_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/"
        inference_data_path = "/workspaces/master-thesis-recurrent-health-events-prediction/data/avh-data-preprocessed/multiple_relapses_patients/"

    model_config_exists = check_if_file_exists(model_config_path)
    training_data_exists = check_if_directory_exists(training_data_path)
    inference_data_exists = check_if_directory_exists(inference_data_path)
    if not model_config_exists:
        raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
    if not training_data_exists:
        raise FileNotFoundError(f"Training data directory not found: {training_data_path}")
    if not inference_data_exists:
        raise FileNotFoundError(f"Inference data directory not found: {inference_data_path}")
    
    if use_fixed_train_test_split:
        neptune_tags.append("fixed_train_test_split")

    main(
        dataset,
        model_config_path=model_config_path,
        training_data_path=training_data_path,
        inference_data_path=inference_data_path,
        skip_survival_dataset_hmm_generation=skip_survival_dataset_hmm_generation,
        neptune_tags=neptune_tags,
        random_state=random_state,
        log_in_neptune=log_in_neptune,
        use_fixed_train_test_split=use_fixed_train_test_split,
    )
