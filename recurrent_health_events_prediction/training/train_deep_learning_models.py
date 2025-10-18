import os
from importlib import resources as impresources
from importlib import import_module
import yaml
from typing import Optional, Tuple
from datetime import datetime
import neptune

import numpy as np
import torch
import torchmetrics
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.training.utils import find_best_threshold, plot_loss_function_epochs
from recurrent_health_events_prediction.training.utils_deep_learning import (
    add_evaluation_results_to_neptune,
    build_or_load_datasets,
    make_tb_writer,
    plot_evaluation_figures,
    preprocess_pair,
)
from recurrent_health_events_prediction.training.utils_traditional_classifier import (
    save_test_predictions,
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config, save_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_model_config_to_neptune,
    add_plotly_plots_to_neptune_run,
    initialize_neptune_run,
    upload_file_to_neptune,
    upload_model_to_neptune,
)


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
    
    has_attention: bool = model.has_attention()

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
            if has_attention:
                outputs_logits, _ = model(x_current, x_past, mask_past)
                outputs_logits = outputs_logits.squeeze(-1)
            else:
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

        
        # Log to TensorBoard
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
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Evaluate the model.

    Args:
        test_dataset (HospReadmDataset): The test dataset.
        model (torch.nn.Module): The trained model.
        batch_size (int): Batch size for evaluation.
    Returns:
        A tuple containing evaluation metrics dictionary, predicted labels,
        predicted probabilities, true labels, and attention scores (if applicable).
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

    all_pred_probs = []
    all_labels = []
    all_attention_scores = []  # For attention models

    with torch.no_grad():
        for batch in test_dataloader:
            x_current, x_past, mask_past, labels = batch
            if model.has_attention():
                outputs_logits, attention_scores = model(x_current, x_past, mask_past)
                outputs_logits = outputs_logits.squeeze(-1)
                attention_scores = attention_scores.squeeze(-1)  # shape (B, T)
                all_attention_scores.append(attention_scores.cpu().numpy())
            else:
                outputs_logits = model(x_current, x_past, mask_past).squeeze(-1)

            probs = torch.sigmoid(outputs_logits).reshape(-1)
            all_pred_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            auroc_metric.update(probs, labels)

    auroc = auroc_metric.compute().item()
    
    # concatenate all batches into one big array (dataset_size, T)
    all_attention_scores = (
        np.concatenate(all_attention_scores, axis=0) if all_attention_scores else None
    )
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    best_threshold, best_f1 = find_best_threshold(all_labels, all_pred_probs)

    all_pred_labels = (all_pred_probs >= best_threshold).astype(int)

    conf_matrix = confusion_matrix(all_labels, all_pred_labels)
    recall = recall_score(all_labels, all_pred_labels)
    accuracy = accuracy_score(all_labels, all_pred_labels)
    precision = precision_score(all_labels, all_pred_labels)

    return {
        "f1_score": best_f1,
        "auroc": auroc,
        "best_threshold": best_threshold,
        "confusion_matrix": conf_matrix,
        "recall": recall,
        "accuracy": accuracy,
        "precision": precision,
    }, all_pred_labels, all_pred_probs, all_labels, all_attention_scores


def prepare_datasets(
    *,
    data_directory: str,
    training_data_config: dict,
    model_config: dict,
    # RAW file names (relative to data_directory) you want to use this time:
    raw_train_filename: str,
    raw_eval_filename: str,
    # Desired .pt filenames (you choose):
    train_pt_name: str,
    eval_pt_name: str,
    cache_pytorch_datasets_path: Optional[str] = None,
    # Last-events:
    need_last_events_eval: bool = False,
    last_events_pt_name: Optional[str] = None,
    # Options:
    save_scaler_dir_path: Optional[str] = None,
    overwrite_preprocessed: bool = False,
    overwrite_pt: bool = False,
) -> Tuple["HospReadmDataset", "HospReadmDataset", Optional["HospReadmDataset"], str, str]:
    """
    Orchestrates preprocessing and dataset caching with fully flexible naming.
    Returns (train_ds, eval_ds, last_ds_or_None, preproc_train_csv, preproc_eval_csv).
    """
    raw_train_csv = os.path.join(data_directory, raw_train_filename)
    raw_eval_csv  = os.path.join(data_directory, raw_eval_filename)

    # 1) produce '<raw>_preprocessed.csv' for each
    preproc_train_csv, preproc_eval_csv = preprocess_pair(
        raw_train_csv=raw_train_csv,
        raw_eval_csv=raw_eval_csv,
        training_data_config=training_data_config,
        save_scaler_dir_path=save_scaler_dir_path,
        output_dir=data_directory,
        overwrite=overwrite_preprocessed,
    )

    # 2) build/load datasets with the .pt names you want
    train_ds, eval_ds, last_ds = build_or_load_datasets(
        preproc_train_csv=preproc_train_csv,
        preproc_eval_csv=preproc_eval_csv,
        model_config=model_config,
        training_data_config=training_data_config,
        cache_dir=cache_pytorch_datasets_path,
        train_pt_name=train_pt_name,
        eval_pt_name=eval_pt_name,
        compute_last_events_eval=need_last_events_eval,
        last_events_pt_name=last_events_pt_name,
        overwrite_pt=overwrite_pt,
    )
    
    print("Datasets are ready.")
    print("Train dataset size:", len(train_ds))
    print("Eval dataset size:", len(eval_ds))
    if need_last_events_eval:
        print("Last-events eval dataset size:", len(last_ds))

    return train_ds, eval_ds, last_ds, preproc_train_csv, preproc_eval_csv


def main(
    model_config_path: str,
    save_scaler_dir_path: str,
    overwrite_data_files: bool = False,
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
    model_name = model_config.get("model_name", "unknown_model")
    model_dir_name = os.path.basename(model_config_dir_path)

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
    train_dataset, test_dataset, last_events_test_dataset, _, _ = prepare_datasets(
        data_directory=data_directory,
        training_data_config=training_data_config,
        model_config=model_config,
        # Raw filenames:
        raw_train_filename="train_full_events.csv",
        raw_eval_filename="test_events.csv",
        # Desired .pt filenames:
        train_pt_name="train_full_dataset.pt",
        eval_pt_name="test_dataset.pt",
        cache_pytorch_datasets_path=save_pytorch_datasets_path,
        # Last-events:
        need_last_events_eval=True,
        last_events_pt_name="last_events_dataset.pt",
        # Options:
        save_scaler_dir_path=save_scaler_dir_path,
        overwrite_preprocessed=overwrite_data_files,
        overwrite_pt=overwrite_data_files
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
        log_dir, f"{model_dir_name}_model.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    # Model train metrics
    train_metrics, _, _, _, _ = evaluate(
        train_dataset, model, batch_size=model_config["batch_size"]
    )
    print("Training metrics: ", train_metrics)

    if neptune_run:
        upload_model_to_neptune(neptune_run, model_save_path)
        add_plotly_plots_to_neptune_run(neptune_run, fig, "loss_per_epoch", "train")
        for metric_name, metric_value in train_metrics.items():
            if metric_name != "confusion_matrix":
                neptune_run[f"train/{metric_name}"] = metric_value
    
    # Evaluate model
    eval_results, _, _, _, _ = evaluate(test_dataset, model, batch_size=model_config["batch_size"])
    eval_results_last_events, all_pred_labels, all_pred_probs, all_labels, all_attention_scores = evaluate(
        last_events_test_dataset, model, batch_size=model_config["batch_size"]
    )

    class_names=training_data_config.get(
            "class_names", ["No Readmission", "Readmission"]
    )

    class_names_dict = {i: class_names[i] for i in range(len(class_names))} if class_names else None

    _ = plot_evaluation_figures(
        all_labels,
        all_pred_probs,
        model_name,
        log_dir,
        class_names_dict=class_names_dict,
        neptune_run=neptune_run,
    )

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
        id_series=last_events_test_dataset.sample_ids,
        y_true=all_labels,
        proba_dict={model_dir_name: all_pred_probs},
        pred_dict={model_dir_name: all_pred_labels},
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
    model_dir_name = "gru"
    multiple_hosp_patients = True  # True if patients can have multiple hospital admissions
    save_scaler_dir_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/scalers"
    if multiple_hosp_patients:
        save_scaler_dir_path += "/multiple_hosp_patients"
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_dir_name}/{model_dir_name}_config.yaml"
    overwrite_preprocessed = False

    LOG_IN_NEPTUNE = False  # Set to True to log in Neptune
    neptune_run_name = f"{model_dir_name}_run"
    # neptune_tags = ["deep_learning", "all_patients", "mimic"]
    neptune_tags = ["deep_learning", "multiple_hosp_patients", "mimic"]
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_run_name = f"{model_dir_name}_{now_str}"  # for TensorBoard logging

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
        overwrite_data_files=overwrite_preprocessed,
        log_in_neptune=LOG_IN_NEPTUNE,
        neptune_run_name=neptune_run_name,
        neptune_tags=neptune_tags,
        tensorboard_run_name=tensorboard_run_name,
    )
