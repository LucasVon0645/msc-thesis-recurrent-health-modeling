from datetime import datetime
import os
import yaml
from importlib import resources as impresources

import optuna
import torch

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.model_selection.deep_learning.utils import save_space_to_txt, save_study_artifacts
from recurrent_health_events_prediction.training.train_deep_learning_models import (
    prepare_datasets,
    train,
    evaluate
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config
import copy

from recurrent_health_events_prediction.utils.neptune_utils import initialize_neptune_run, upload_file_to_neptune

# ===========================================================
# 1. Global setup
# ===========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CONFIG_PATH = (impresources.files(configs) / "data_config.yaml")
MODEL_NAME = "gru"
MODEL_BASE_CONFIG_PATH = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{MODEL_NAME}/{MODEL_NAME}_config.yaml"
SAVE_SCALER_DIR = "/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/scalers/multiple_hosp_patients"
CACHE_PYTORCH_DATASETS_PATH = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{MODEL_NAME}/multiple_hosp_patients"
N_TRIALS = 50
LOG_IN_NEPTUNE = True  # Set to True to log results to Neptune.ai

base_config = import_yaml_config(MODEL_BASE_CONFIG_PATH)
with open(DATA_CONFIG_PATH) as f:
    data_config = yaml.safe_load(f)
training_data_config = data_config["training_data"]["mimic"]
data_directory = training_data_config["data_directory"]

# Preload datasets once (saves time per trial)
train_dataset, validation_dataset, _, preproc_train_csv, preproc_eval_csv = prepare_datasets(
    data_directory=data_directory,
    training_data_config=training_data_config,
    model_config=base_config,
    # Raw filenames:
    raw_train_filename="train_events.csv",
    raw_eval_filename="validation_events.csv",
    # Desired .pt filenames:
    train_pt_name="train_dataset.pt",
    eval_pt_name="validation_dataset.pt",
    cache_pytorch_datasets_path=CACHE_PYTORCH_DATASETS_PATH,
    # Last-events:
    need_last_events_eval=False,
    last_events_pt_name="last_events_dataset.pt",
    # Options:
    save_scaler_dir_path=SAVE_SCALER_DIR,
    overwrite_preprocessed=False,
    overwrite_pt=False
)

# ===========================================================
# 2. Define model-specific hyperparameter spaces
# ===========================================================

space_hyperparams = {
    # Common hyperparameters
    "learning_rate": (1e-4, 1e-2),  # log-uniform
    "batch_size": [32, 64, 128],  # categorical
    "dropout": (0.0, 0.5),  # uniform
    "hidden_size_head": [8, 16, 32, 64, 128],  # categorical
}

def sample_hparams(trial, model_class_name):
    """Return a dictionary of sampled hyperparameters for the given model class."""
    params = {}

    # Common hyperparameters
    learning_rate_interval = space_hyperparams["learning_rate"]
    params["learning_rate"] = trial.suggest_loguniform("learning_rate", *learning_rate_interval)
    batch_size_values = space_hyperparams["batch_size"]
    params["batch_size"] = trial.suggest_categorical("batch_size", batch_size_values)
    dropout_interval = space_hyperparams["dropout"]
    params["dropout"] = trial.suggest_float("dropout", *dropout_interval)
    hidden_size_head_values = space_hyperparams["hidden_size_head"]
    params["hidden_size_head"] = trial.suggest_categorical("hidden_size_head", hidden_size_head_values)

    # Model-specific parameters
    if model_class_name == "GRUNet":
        hidden_size_seq_values = [16, 32, 64, 128, 256]
        params["hidden_size_seq"] = trial.suggest_categorical("hidden_size_seq", hidden_size_seq_values)
        num_layers_seq_values = [1, 2]
        params["num_layers_seq"] = trial.suggest_categorical("num_layers_seq", num_layers_seq_values)
        
        if "hidden_size_seq" not in space_hyperparams.keys():
            space_hyperparams["hidden_size_seq"] = hidden_size_seq_values
        if "num_layers_seq" not in space_hyperparams.keys():
            space_hyperparams["num_layers_seq"] = num_layers_seq_values

    elif model_class_name == "AttentionPoolingNet":
        hidden_size_seq_values = [8, 16, 32, 64, 128]
        params["hidden_size_seq"] = trial.suggest_categorical("hidden_size_seq", hidden_size_seq_values)
        if "hidden_size_seq" not in space_hyperparams.keys():
            space_hyperparams["hidden_size_seq"] = hidden_size_seq_values
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    return params


# ===========================================================
# 3. Objective function
# ===========================================================

def objective(trial, model_class_name):
    # Deep copy base config to modify safely
    model_config = copy.deepcopy(base_config)
    sampled = sample_hparams(trial, model_class_name)

    # Inject into model_config
    model_config["learning_rate"] = sampled["learning_rate"]
    model_config["batch_size"] = sampled["batch_size"]
    model_config["model_params"]["hidden_size_head"] = sampled["hidden_size_head"]
    model_config["model_params"]["dropout"] = sampled["dropout"]

    if model_class_name == "GRUNet":
        model_config["model_params"]["hidden_size_seq"] = sampled["hidden_size_seq"]
        model_config["model_params"]["num_layers_seq"] = sampled["num_layers_seq"]
        model_config["model_class"] = "GRUNet"
    else:
        model_config["model_params"]["hidden_size_seq"] = sampled["hidden_size_seq"]
        model_config["model_class"] = "AttentionPoolingNet"

    # Train
    model, _ = train(train_dataset, model_config)

    # Evaluate on test (you can also split train into train/val if you prefer)
    results, _, _, _, _ = evaluate(validation_dataset, model, batch_size=model_config["batch_size"])

    trial.set_user_attr("auroc", results["auroc"])
    trial.set_user_attr("f1", results["f1_score"])
    print(f"Trial {trial.number} - AUROC: {results['auroc']:.4f}, F1: {results['f1_score']:.4f}")

    # Optuna maximizes the return value, so choose AUROC or F1
    return results["auroc"]


# ===========================================================
# 4. Run the study
# ===========================================================

def run_study(model_class_name="GRUNet", n_trials=10):
    print(f"Starting Optuna study for {model_class_name} with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name=f"{model_class_name}_tuning")
    study.optimize(lambda trial: objective(trial, model_class_name), n_trials=n_trials)
    print(f"Best {model_class_name} trial:")
    print(study.best_trial.params)
    print(f"Best F1: {study.best_value:.4f}")
    return study


if __name__ == "__main__":
    model_class_name = base_config.get("model_class", "GRUNet")

    # Initialize Neptune run
    neptune_run_name = f"{MODEL_NAME}_hparam_tuning"
    neptune_tags = [MODEL_NAME, "hparam_tuning", "optuna"]
    neptune_run = (
        initialize_neptune_run(
            DATA_CONFIG_PATH, neptune_run_name, "mimic", tags=neptune_tags
        )
        if LOG_IN_NEPTUNE
        else None
    )

    
    print(f"\n===== Tuning {model_class_name} =====")
    print("Base config from model name:", MODEL_NAME)
    
    # Set up Optuna logging directory
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    optuna_run_name = f"{MODEL_NAME}_optuna_{now_str}"  # for TensorBoard logging
    base_log_dir = training_data_config.get("optuna_log_dir", "_optuna_runs")
    save_dir = os.path.join(base_log_dir, optuna_run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Optuna study artifacts will be saved to: {save_dir}\n")
    if neptune_run:
        neptune_run["study_log"] = optuna_run_name
        neptune_run["data/train_size"] = len(train_dataset)
        neptune_run["data/val_size"] = len(validation_dataset)
        neptune_run["data/train_path"] = str(preproc_train_csv)
        neptune_run["data/val_path"] = str(preproc_eval_csv)
        
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=MODEL_BASE_CONFIG_PATH,
            neptune_base_path="artifacts",
            neptune_filename="base_config.yaml"
        )
    
    # Run the study
    study = run_study(model_class_name, n_trials=N_TRIALS)
    
    filepath = save_space_to_txt(space_hyperparams, out_dir=save_dir)
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=filepath,
            neptune_base_path="artifacts",
            neptune_filename="hyperparams_search_space.txt"
        )
    
    # Save study artifacts
    save_study_artifacts(
        study,
        out_dir=save_dir,
        base_config=base_config,
        model_class_name=model_class_name,
        neptune_run=neptune_run
    )
