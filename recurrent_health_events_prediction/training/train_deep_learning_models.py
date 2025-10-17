import os
from importlib import resources as impresources
from importlib import import_module
import yaml
from typing import Optional, Tuple
from datetime import datetime
import neptune

import pandas as pd
import numpy as np
import torch
import torchmetrics
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, recall_score, accuracy_score, precision_score

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.training.utils import find_best_threshold, plot_loss_function_epochs, plot_pred_proba_distribution, standard_scale_data
from recurrent_health_events_prediction.training.utils import plot_confusion_matrix
from recurrent_health_events_prediction.training.utils_traditional_classifier import save_test_predictions
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config, save_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_model_config_to_neptune,
    add_plot_to_neptune_run,
    add_plotly_plots_to_neptune_run,
    initialize_neptune_run,
    upload_file_to_neptune,
    upload_model_to_neptune,
)

def add_evaluation_results_to_neptune(
    neptune_run: neptune.Run,
    eval_results: dict,
    class_names: Optional[list[str]] = None,
    last_events: bool = False,
    
):
    neptune_path = "evaluation/last_events" if last_events else "evaluation"
    for metric_name, metric_value in eval_results.items():
        if metric_name == "confusion_matrix":
            fig = plot_confusion_matrix(conf_matrix=metric_value, class_names=class_names)
            add_plot_to_neptune_run(neptune_run, "confusion_matrix", fig, neptune_path)
        else:
            neptune_run[f"{neptune_path}/{metric_name}"] = metric_value


def scale_preprocessed_data(
    data_directory: str, training_data_config: dict, save_scaler_dir_path: str = None
):
    """
    Scale preprocessed data for training and testing.

    Args:
        data_directory (str): Directory containing the dataset.
        training_config (dict): Model configuration parameters.
    Returns:
        Tuple containing preprocessed training and testing data file paths.
    """
    train_file_path = os.path.join(data_directory, "train_events.csv")
    test_file_path = os.path.join(data_directory, "test_events.csv")

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # Preprocess
    features_to_scale = training_data_config["features_to_scale"]

    train_scaled_df, test_scaled_df = standard_scale_data(
        train_df, test_df, features_to_scale, save_scaler_dir_path=save_scaler_dir_path
    )

    train_preprocessed_file_path = os.path.join(
        data_directory, "train_events_preprocessed.csv"
    )
    test_preprocessed_file_path = os.path.join(
        data_directory, "test_events_preprocessed.csv"
    )

    train_scaled_df.to_csv(train_preprocessed_file_path, index=False)
    test_scaled_df.to_csv(test_preprocessed_file_path, index=False)

    return train_preprocessed_file_path, test_preprocessed_file_path


def get_train_test_datasets(
    train_df_path, test_df_path, model_config, training_data_config
):
    """
    Get training and testing datasets.
    Args:
        train_df_path (str): Path to the training data CSV file.
        test_df_path (str): Path to the testing data CSV file.
        model_config (dict): Model configuration parameters.
        training_data_config (dict): Training configuration parameters.
    Returns:
        Tuple containing training and testing datasets.
    """

    dataset_config = {
        "longitudinal_feat_cols": model_config["longitudinal_feat_cols"],
        "current_feat_cols": model_config["current_feat_cols"],
        "max_seq_len": model_config.get("max_sequence_length", 5),
        "no_elective": model_config.get("no_elective", True),
        "reverse_chronological_order": model_config.get(
            "reverse_chronological_order", True
        ),
        # column names config
        "subject_id_col": training_data_config.get("patient_id_col", "SUBJECT_ID"),
        "order_col": training_data_config.get("time_col", "ADMITTIME"),
        "label_col": training_data_config.get(
            "binary_event_col", "READMISSION_30_DAYS"
        ),
        "next_admt_type_col": training_data_config.get(
            "next_admt_type_col", "NEXT_ADMISSION_TYPE"
        ),
        "hosp_id_col": training_data_config.get("hosp_id_col", "HADM_ID"),
    }

    # Create datasets
    train_dataset = HospReadmDataset(csv_path=train_df_path, **dataset_config)

    test_dataset = HospReadmDataset(csv_path=test_df_path, **dataset_config)

    return train_dataset, test_dataset

def get_test_last_events_only_dataset(test_df_path, model_config, training_data_config):
    """
    Get testing dataset with only the last event per patient.
    Args:
        test_df_path (str): Path to the testing data CSV file.
        model_config (dict): Model configuration parameters.
        training_data_config (dict): Training configuration parameters.
    Returns:
        Testing dataset with only the last event per patient.
    """

    dataset_config = {
        "longitudinal_feat_cols": model_config["longitudinal_feat_cols"],
        "current_feat_cols": model_config["current_feat_cols"],
        "max_seq_len": model_config.get("max_sequence_length", 5),
        "no_elective": model_config.get("no_elective", True),
        "reverse_chronological_order": model_config.get(
            "reverse_chronological_order", True
        ),
        "last_events_only": True,  # Only keep last event per patient
        # column names config
        "subject_id_col": training_data_config.get("patient_id_col", "SUBJECT_ID"),
        "order_col": training_data_config.get("time_col", "ADMITTIME"),
        "label_col": training_data_config.get(
            "binary_event_col", "READMISSION_30_DAYS"
        ),
        "next_admt_type_col": training_data_config.get(
            "next_admt_type_col", "NEXT_ADMISSION_TYPE"
        ),
        "hosp_id_col": training_data_config.get("hosp_id_col", "HADM_ID"),
    }

    # Create dataset
    test_dataset = HospReadmDataset(csv_path=test_df_path, **dataset_config)

    return test_dataset


def train(
    train_dataset: HospReadmDataset,
    model_config: dict,
    ModelClass: Optional[torch.nn.Module] = None,
    neptune_run: Optional[neptune.Run] = None,
    writer = None
) -> Tuple[torch.nn.Module, list[float]]:
    """
    Train the model.

    Args:
        train_dataset (HospReadmDataset): Training dataset.
        model_config (dict): Model configuration parameters.
    Returns:
        Trained model and training loss history.
    """

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )

    # Initialize model
    if ModelClass is not None:
        print(f"\nUsing provided model class: {ModelClass.__name__}")
    else:
        model_class_name = model_config.get("model_class", "GRUNet")

        # Class resolver: assume `model_class_name` is a class defined in
        # `recurrent_health_events_prediction.model.RecurrentHealthEventsDL`.
        mod = import_module(
            "recurrent_health_events_prediction.model.RecurrentHealthEventsDL"
        )
        try:
            ModelClass = getattr(mod, model_class_name)
            print(f"\nUsing model class: {model_class_name}")
        except AttributeError:
            raise ImportError(
                f"Model class '{model_class_name}' not found in RecurrentHealthEventsDL"
            )

    model: torch.nn.Module = ModelClass(**model_config["model_params"])
    
    # once, right after you create `train_loader` and move `model` to device
    if writer is not None:
        model.eval()
        with torch.no_grad():
            x_current, x_past, mask_past, _ = next(iter(train_loader))
            device = next(model.parameters()).device
            x_current, x_past, mask_past = (x_current.to(device), x_past.to(device), mask_past.to(device))
            try:
                writer.add_graph(model, (x_current, x_past, mask_past))
            except Exception as e:
                print(f"Skipping add_graph due to: {e}")
        model.train()


    print("\nModel initialized and ready for training.")
    print("Model parameters:")
    for key, value in model_config["model_params"].items():
        print(f"  {key}: {value}")

    # Set up loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])

    model.train()
    loss_over_epochs = []

    print("\nStarting training...")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Learning rate: {model_config['learning_rate']}")
    print("Optimizer: Adam")
    print("Loss function: BCEWithLogitsLoss\n")

    # Training loop
    for epoch in range(model_config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x_current, x_past, mask_past, labels = batch
            optimizer.zero_grad()
            outputs_logits = model(x_current, x_past, mask_past).squeeze(-1)
            loss = criterion(outputs_logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        loss_over_epochs.append(avg_epoch_loss)
        print(
            f"Epoch {epoch + 1}/{model_config['num_epochs']}, Loss: {avg_epoch_loss}"
        )
        if neptune_run:
            neptune_run["train/loss"].log(avg_epoch_loss)

        
        # Log to TensorBoard (new)
        if writer is not None:
            writer.add_scalar("train/loss", avg_epoch_loss, epoch + 1)

            # Weights & grad histograms (per epoch)
            for name, param in model.named_parameters():
                if param.requires_grad and param.data is not None:
                    writer.add_histogram(f"weights/{name}", param.data, epoch + 1)
                if param.grad is not None:
                    writer.add_histogram(f"grads/{name}", param.grad, epoch + 1)

    print("\nTraining complete.")
    
    

    return model, loss_over_epochs


def evaluate(
    test_dataset: HospReadmDataset,
    model: torch.nn.Module,
    batch_size: 64,
) -> dict:
    """
    Evaluate the model.

    Args:
        test_dataset (HospReadmDataset): The test dataset.
        model (torch.nn.Module): The trained model.
    """

    print("Starting evaluation...")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}\n")

    model.eval()
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Evaluation metrics
    auroc_metric = torchmetrics.AUROC(task="binary", num_classes=2)

    all_pred_probs = np.array([])
    all_labels = np.array([])

    with torch.no_grad():
        for batch in test_dataloader:
            x_current, x_past, mask_past, labels = batch
            outputs_logits = model(x_current, x_past, mask_past)
            probs = torch.sigmoid(outputs_logits)
            all_pred_probs = np.concatenate((all_pred_probs, probs.cpu().numpy()))
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()))

            auroc_metric.update(probs, labels)

    auroc = auroc_metric.compute().item()

    best_threshold, best_f1 = find_best_threshold(all_labels, all_pred_probs)

    all_pred_labels = (all_pred_probs >= best_threshold).astype(int)

    conf_matrix = confusion_matrix(all_labels, all_pred_labels)
    recall = recall_score(all_labels, all_pred_labels)
    accuracy = accuracy_score(all_labels, all_pred_labels)
    precision = precision_score(all_labels, all_pred_labels)

    print(f"Evaluation results - AUROC: {auroc}, Best F1-Score: {best_f1}, Best Threshold: {best_threshold}")

    return {
        "f1_score": best_f1,
        "auroc": auroc,
        "best_threshold": best_threshold,
        "confusion_matrix": conf_matrix,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }, all_pred_labels, all_pred_probs, all_labels


def make_tb_writer(log_dir: str | None = None):
    # e.g., logs go under runs/gru_model_run/<timestamp>
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def prepare_train_test_datasets(
    data_directory: str,
    training_data_config: dict,
    model_config: dict,
    save_output_dir_path: str,
    save_scaler_dir_path: str | None = None,
    overwrite_preprocessed: bool = False,
) -> tuple:
    """
    Load or create preprocessed train/test CSVs and corresponding PyTorch datasets.

    Returns:
        train_dataset, test_dataset, train_df_path, test_df_path
    """
    print("Preparing train and test datasets...")
    # Paths to preprocessed CSVs
    train_df_path = os.path.join(data_directory, "train_events_preprocessed.csv")
    test_df_path = os.path.join(data_directory, "test_events_preprocessed.csv")

    if (
        not os.path.exists(train_df_path)
        or not os.path.exists(test_df_path)
        or overwrite_preprocessed
    ):
        train_df_path, test_df_path = scale_preprocessed_data(
            data_directory,
            training_data_config,
            save_scaler_dir_path=save_scaler_dir_path,
        )

    pytorch_train_dataset_path = os.path.join(save_output_dir_path, "train_dataset.pt")
    pytorch_test_dataset_path = os.path.join(save_output_dir_path, "test_dataset.pt")
    pytorch_last_events_test_dataset_path = os.path.join(save_output_dir_path, "last_events_test_dataset.pt")

    if (
        os.path.exists(pytorch_train_dataset_path)
        and os.path.exists(pytorch_test_dataset_path)
        and os.path.exists(pytorch_last_events_test_dataset_path)
        and not overwrite_preprocessed
    ):
        print("Loading existing PyTorch datasets...")
        train_dataset = torch.load(pytorch_train_dataset_path, weights_only=False)
        test_dataset = torch.load(pytorch_test_dataset_path, weights_only=False)
        last_events_test_dataset = torch.load(pytorch_last_events_test_dataset_path, weights_only=False)
        print(f"Test dataset (all events) size: {len(test_dataset)}")
        print(f"Test dataset (last events only) size: {len(last_events_test_dataset)}")
    else:
        print("Creating new PyTorch datasets...")

        train_dataset, test_dataset = get_train_test_datasets(
            train_df_path,
            test_df_path,
            model_config,
            training_data_config=training_data_config,
        )
        
        last_events_test_dataset = get_test_last_events_only_dataset(
            test_df_path,
            model_config,
            training_data_config=training_data_config,
        )
        print(f"Test dataset (all events) size: {len(test_dataset)}")
        print(f"Test dataset (last events only) size: {len(last_events_test_dataset)}")

        # Ensure target directory exists
        os.makedirs(save_output_dir_path, exist_ok=True)

        torch.save(train_dataset, pytorch_train_dataset_path)
        torch.save(test_dataset, pytorch_test_dataset_path)
        torch.save(last_events_test_dataset, pytorch_last_events_test_dataset_path)

        print(f"Saved PyTorch datasets to {save_output_dir_path}")

    return train_dataset, test_dataset, last_events_test_dataset, train_df_path, test_df_path


def main(
    model_config_path: str,
    save_scaler_dir_path: str,
    overwrite_preprocessed: bool = False,
    save_pytorch_datasets_path: Optional[str] = None,
    log_in_neptune: bool = False,
    neptune_tags: Optional[list[str]] = None,
    neptune_run_name: str = "deep_learning_model_run",
    tensorboard_run_name: str = "deep_learning_model_run",
):
    print("Starting training script...")
    print("Logging in Neptune:", log_in_neptune)

    data_config_path = (impresources.files(configs) / "data_config.yaml")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    training_data_config = data_config["training_data"]["mimic"]
    data_directory = training_data_config["data_directory"]

    model_config = import_yaml_config(model_config_path)
    model_config_dir_path = os.path.dirname(model_config_path)
    model_name = os.path.basename(model_config_dir_path)

    print(f"Using model config from: {model_config_path}")
    print(f"Model name: {model_name}")
    print(f"Data directory: {data_directory}")

    model_params_dict = model_config['model_params']
    assert model_params_dict["input_size_curr"] == len(
        model_config["current_feat_cols"]
    ), f"mismatch in input_size_curr and current_feat_cols length"
    assert model_params_dict["input_size_seq"] == len(
        model_config["longitudinal_feat_cols"]
    ), f"mismatch in input_size_seq and longitudinal_feat_cols length"

    neptune_run = (
        initialize_neptune_run(
            data_config_path, neptune_run_name, "mimic", tags=neptune_tags
        )
        if log_in_neptune
        else None
    )

    if neptune_run:
        add_model_config_to_neptune(neptune_run, model_config)
        neptune_run["train/data_directory"] = data_directory
        neptune_run["tensorboard/run_name"] = tensorboard_run_name

    if save_pytorch_datasets_path is None:
        save_pytorch_datasets_path = model_config_dir_path

    # Prepare datasets (load existing or create + save new)
    train_dataset, test_dataset, last_events_test_dataset, _, _ = prepare_train_test_datasets(
        data_directory=data_directory,
        training_data_config=training_data_config,
        model_config=model_config,
        save_output_dir_path=save_pytorch_datasets_path,
        save_scaler_dir_path=save_scaler_dir_path,
        overwrite_preprocessed=overwrite_preprocessed,
    )

    print("Initializing TensorBoard writer...")
    # Create TensorBoard writer (log dir name matches your run name)
    base_log_dir = training_data_config.get("tensorboard_log_dir", "_runs")
    log_dir = os.path.join(base_log_dir, tensorboard_run_name)
    writer = make_tb_writer(log_dir=log_dir)

    # Save model_config as YAML in log_dir
    model_config_yaml_outpath = os.path.join(log_dir, "model_config.yaml")
    save_yaml_config(model_config, model_config_yaml_outpath)
    print(f"Saved model config to {model_config_yaml_outpath}")

    # Train model

    model, loss_epochs = train(
        train_dataset, model_config, neptune_run=neptune_run, writer=writer
    )

    fig = plot_loss_function_epochs(
        loss_epochs,
        num_samples=len(train_dataset),
        batch_size=model_config["batch_size"],
        learning_rate=model_config["learning_rate"],
        save_fig_dir=log_dir,
    )

    print("Saving trained model...")
    # Save the trained model
    model_save_path = os.path.join(
        model_config_dir_path, f"{model_name}_model.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    if neptune_run:
        upload_model_to_neptune(neptune_run, model_save_path)
        add_plotly_plots_to_neptune_run(neptune_run, fig, "loss_per_epoch", "train")

    eval_results, _, _, _ = evaluate(test_dataset, model, batch_size=model_config["batch_size"])
    eval_results_last_events, all_pred_labels, all_pred_probs, all_labels = evaluate(
        last_events_test_dataset, model, batch_size=model_config["batch_size"]
    )
    
    class_names=training_data_config.get(
            "class_names", ["No Readmission", "Readmission"]
    )

    fig = plot_pred_proba_distribution(
        all_pred_labels,
        all_pred_probs,
        show_plot=False,
        save_dir_path=log_dir,
        class_names=class_names
    )
    if neptune_run:
        add_plot_to_neptune_run(neptune_run, "pred_proba_distribution", fig, neptune_path="evaluation/last_events")

    print("Evaluation on all test events:", eval_results)
    print("Evaluation on last test events only:", eval_results_last_events)

    # Log eval scalars to TensorBoard
    for k, v in eval_results.items():
        if k != "confusion_matrix":  # Skip confusion matrix for scalar logging
            writer.add_scalar(f"eval/{k}", v)

    writer.flush()
    writer.close()

    # ===== Save All Predictions Last Events =====
    pred_test_output_filepath = os.path.join(log_dir, "test_predictions.csv")
    print("\nSaving test predictions...")
    print("Output path for test predictions: ", pred_test_output_filepath)
    save_test_predictions(
        out_path=pred_test_output_filepath,
        id_series=[0]*len(all_labels),  # Placeholder if no specific ID is needed
        y_true=all_labels,
        proba_dict={model_name: all_pred_probs},
        pred_dict={model_name: all_pred_labels},
        file_format="csv"  # Change to "parquet" if needed
    )

    if neptune_run:
        add_evaluation_results_to_neptune(
            neptune_run,
            eval_results,
            class_names=class_names,
            last_events=False,
        )
        add_evaluation_results_to_neptune(
            neptune_run,
            eval_results_last_events,
            class_names=class_names,
            last_events=True,
        )
        upload_file_to_neptune(
            neptune_run,
            pred_test_output_filepath,
            neptune_base_path="artifacts/inference",
            neptune_filename="test_predictions.csv"
        )
        upload_file_to_neptune(
            neptune_run,
            model_config_yaml_outpath,
            neptune_base_path="artifacts/config",
            neptune_filename="model_config.yaml"
        )
        neptune_run["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        neptune_run["training_data_stats/train/num_samples"] = len(train_dataset)
        neptune_run["training_data_stats/test/num_samples"] = len(test_dataset)
        neptune_run["training_data_stats/test/num_samples_last_events"] = len(last_events_test_dataset)
        neptune_run.stop()

    print("Training and evaluation complete.")


if __name__ == "__main__":
    print("Imports complete. Running main...")
    model_name = "gru_duration_aware_min"
    multiple_hosp_patients = True  # True if patients can have multiple hospital admissions
    save_scaler_dir_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/scalers"
    if multiple_hosp_patients:
        save_scaler_dir_path += "/multiple_hosp_patients"
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_name}/{model_name}_config.yaml"
    overwrite_preprocessed = True

    LOG_IN_NEPTUNE = False  # Set to True to log in Neptune
    neptune_run_name = f"{model_name}_run"
    # neptune_tags = ["deep_learning", "all_patients", "mimic"]
    neptune_tags = ["deep_learning", "multiple_hosp_patients", "mimic"]
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_run_name = f"{model_name}_{now_str}"  # for TensorBoard logging

    # Check if the provided paths exist
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
    if not os.path.exists(save_scaler_dir_path):
        raise FileNotFoundError(f"Scaler directory path does not exist: {save_scaler_dir_path}")
    
    if multiple_hosp_patients:
        save_pytorch_datasets_path = os.path.dirname(model_config_path) + "/multiple_hosp_patients"
    else:
        save_pytorch_datasets_path = None  # Saves in the same dir as model_config_path

    main(
        model_config_path=model_config_path,
        save_scaler_dir_path=save_scaler_dir_path,
        save_pytorch_datasets_path=save_pytorch_datasets_path,
        overwrite_preprocessed=overwrite_preprocessed,
        log_in_neptune=LOG_IN_NEPTUNE,
        neptune_run_name=neptune_run_name,
        neptune_tags=neptune_tags,
        tensorboard_run_name=tensorboard_run_name,
    )
