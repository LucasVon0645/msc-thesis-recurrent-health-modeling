import os
from importlib import resources as impresources
from importlib import import_module
import yaml
from typing import Optional, Tuple

import neptune
import pandas as pd
import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter


from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.preprocessing.utils import (
    remap_discharge_location,
    remap_mimic_races,
)
from recurrent_health_events_prediction.training.utils import (
    preprocess_features_to_one_hot_encode,
    standard_scale_data,
)
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config
from recurrent_health_events_prediction.utils.neptune_utils import add_model_config_to_neptune, initialize_neptune_run, upload_model_to_neptune


def create_preprocessed_data(
    data_directory: str, training_data_config: dict, save_scaler_dir_path: str = None
):
    """
    Load and preprocess data for training and testing.

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

    train_df = remap_discharge_location(train_df)
    train_df = remap_mimic_races(train_df)
    test_df = remap_discharge_location(test_df)
    test_df = remap_mimic_races(test_df)

    # Preprocess
    features_to_scale = training_data_config["features_to_scale"]
    features_to_one_hot_encode = training_data_config["features_to_one_hot_encode"]
    one_hot_cols_to_drop = training_data_config["one_hot_cols_to_drop"]

    train_scaled_df, test_scaled_df = standard_scale_data(
        train_df, test_df, features_to_scale, save_scaler_dir_path=save_scaler_dir_path
    )

    train_preprocessed_df, _ = preprocess_features_to_one_hot_encode(
        train_scaled_df,
        features_to_encode=features_to_one_hot_encode,
        one_hot_cols_to_drop=one_hot_cols_to_drop,
    )

    test_preprocessed_df, _ = preprocess_features_to_one_hot_encode(
        test_scaled_df,
        features_to_encode=features_to_one_hot_encode,
        one_hot_cols_to_drop=one_hot_cols_to_drop,
    )

    train_preprocessed_file_path = os.path.join(
        data_directory, "train_events_preprocessed.csv"
    )
    test_preprocessed_file_path = os.path.join(
        data_directory, "test_events_preprocessed.csv"
    )

    train_preprocessed_df.to_csv(train_preprocessed_file_path, index=False)
    test_preprocessed_df.to_csv(test_preprocessed_file_path, index=False)

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
        "next_admt_type_col": model_config.get(
            "next_admt_type_col", "NEXT_ADMISSION_TYPE"
        ),
        "hosp_id_col": model_config.get("hosp_id_col", "HADM_ID"),
    }

    # Create datasets
    train_dataset = HospReadmDataset(csv_path=train_df_path, **dataset_config)

    test_dataset = HospReadmDataset(csv_path=test_df_path, **dataset_config)

    return train_dataset, test_dataset


def train(
    train_dataset: HospReadmDataset,
    model_config: dict,
    ModelClass: Optional[torch.nn.Module] = None,
    neptune_run: Optional[neptune.Run] = None,
    writer: Optional[SummaryWriter] = None
) -> Tuple[torch.nn.Module, list[float]]:
    """
    Train the model.

    Args:
        train_dataset (HospReadmDataset): Training dataset.
        model_config (dict): Model configuration parameters.
    Returns:
        Trained model and training loss history.
    """

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )
    
    if writer is not None:
        sample_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_config["batch_size"], shuffle=True)
        x_current, x_past, mask_past, _ = next(iter(sample_loader))
        try:
            writer.add_graph(model, (x_current, x_past, mask_past))
        except Exception as e:
            print(f"Skipping add_graph due to: {e}")

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
        for batch in train_dataloader:
            x_current, x_past, mask_past, labels = batch
            optimizer.zero_grad()
            outputs_logits = model(x_current, x_past, mask_past).squeeze(-1)
            loss = criterion(outputs_logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        loss_over_epochs.append(avg_epoch_loss)
        print(
            f"Epoch {epoch + 1}/{model_config['num_epochs']}, Loss: {avg_epoch_loss}"
        )
        if neptune_run:
            neptune_run.log_metric("train/loss", avg_epoch_loss)
        
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
    prob_threshold: float = 0.5,
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
    accuracy_metric = torchmetrics.Accuracy(task="binary", num_classes=2)
    f1_metric = torchmetrics.F1Score(task="binary", num_classes=2)
    auroc_metric = torchmetrics.AUROC(task="binary", num_classes=2)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            x_current, x_past, mask_past, labels = batch
            outputs_logits = model(x_current, x_past, mask_past)
            probs = torch.sigmoid(outputs_logits)
            preds = (probs >= prob_threshold).long().squeeze(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            auroc_metric.update(torch.sigmoid(outputs_logits), labels)

    accuracy = accuracy_metric.compute().item()
    f1 = f1_metric.compute().item()
    auroc = auroc_metric.compute().item()

    print(f"Evaluation results - Accuracy: {accuracy}, F1 Score: {f1}, AUROC: {auroc}")

    return {"accuracy": accuracy, "f1_score": f1, "auroc": auroc}

def make_tb_writer(log_dir: str | None = None) -> SummaryWriter:
    # e.g., logs go under runs/gru_model_run/<timestamp>
    return SummaryWriter(log_dir=log_dir)

def prepare_train_test_datasets(
    data_directory: str,
    training_data_config: dict,
    model_config: dict,
    model_config_dir_path: str,
    save_scaler_dir_path: str | None = None,
    overwrite_preprocessed: bool = False,
) -> tuple:
    """
    Load or create preprocessed train/test CSVs and corresponding PyTorch datasets.

    Returns:
        train_dataset, test_dataset, train_df_path, test_df_path
    """

    # Paths to preprocessed CSVs
    train_df_path = os.path.join(data_directory, "train_events_preprocessed.csv")
    test_df_path = os.path.join(data_directory, "test_events_preprocessed.csv")

    if (
        not os.path.exists(train_df_path)
        or not os.path.exists(test_df_path)
        or overwrite_preprocessed
    ):
        train_df_path, test_df_path = create_preprocessed_data(
            data_directory,
            training_data_config,
            save_scaler_dir_path=save_scaler_dir_path,
        )

    pytorch_train_dataset_path = os.path.join(model_config_dir_path, "train_dataset.pt")
    pytorch_test_dataset_path = os.path.join(model_config_dir_path, "test_dataset.pt")

    if (
        os.path.exists(pytorch_train_dataset_path)
        and os.path.exists(pytorch_test_dataset_path)
        and not overwrite_preprocessed
    ):
        print("Loading existing PyTorch datasets...")
        train_dataset = torch.load(pytorch_train_dataset_path)
        test_dataset = torch.load(pytorch_test_dataset_path)
    else:
        print("Creating new PyTorch datasets...")

        train_dataset, test_dataset = get_train_test_datasets(
            train_df_path,
            test_df_path,
            model_config,
            training_data_config=training_data_config,
        )

        # Ensure target directory exists
        os.makedirs(model_config_dir_path, exist_ok=True)

        torch.save(train_dataset, pytorch_train_dataset_path)
        torch.save(test_dataset, pytorch_test_dataset_path)

        print(f"Saved PyTorch datasets to {model_config_dir_path}")

    return train_dataset, test_dataset, train_df_path, test_df_path

def main(
    model_config_path: str,
    save_scaler_dir_path: str,
    overwrite_preprocessed: bool = False,
    log_in_neptune: bool = False,
    neptune_tags: Optional[list[str]] = None,
    neptune_run_name: str = "deep_learning_model_run",
):
    data_config_path = (impresources.files(configs) / "data_config.yaml")

    with open(data_config_path) as f:
        data_config = yaml.safe_load(f)

    training_data_config = data_config["training_data"]["mimic"]
    data_directory = training_data_config["data_directory"]

    model_config = import_yaml_config(model_config_path)
    model_config_dir_path = os.path.dirname(model_config_path)
    model_name = os.path.basename(model_config_dir_path)

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

    # Prepare datasets (load existing or create + save new)
    train_dataset, test_dataset, _, _ = prepare_train_test_datasets(
        data_directory=data_directory,
        training_data_config=training_data_config,
        model_config=model_config,
        model_config_dir_path=model_config_dir_path,
        save_scaler_dir_path=save_scaler_dir_path,
        overwrite_preprocessed=overwrite_preprocessed,
    )

    # Create TensorBoard writer (log dir name matches your run name)
    base_log_dir = training_data_config.get("tensorboard_log_dir", "_runs")
    writer = make_tb_writer(log_dir=f"{base_log_dir}/{neptune_run_name}")
    
    # Train model

    model, _ = train(train_dataset, model_config, neptune_run=neptune_run, writer=writer)

    # Save the trained model
    model_save_path = os.path.join(
        model_config["model_config_dir_path"], f"{model_name}_model.pth"
    )
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")
    
    if neptune_run:
        upload_model_to_neptune(neptune_run, model_save_path)

    eval_results = evaluate(test_dataset, model, batch_size=model_config["batch_size"])
    
    # Log eval scalars to TensorBoard
    for k, v in eval_results.items():
        writer.add_scalar(f"eval/{k}", v)

    writer.flush()
    writer.close()
    
    if neptune_run:
        for metric_name, metric_value in eval_results.items():
            neptune_run.log_metric(f"eval/{metric_name}", metric_value)
        neptune_run.stop()

    print("Training and evaluation complete.")


if __name__ == "__main__":
    model_name = "gru"
    save_scaler_dir_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/scalers"
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_name}/{model_name}_config.yaml"
    overwrite_preprocessed = False

    LOG_IN_NEPTUNE = True
    neptune_run_name = f"{model_name}_model_run"
    neptune_tags = ["deep_learning", "all_patients", "mimic"]

    # Check if the provided paths exist
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model config path does not exist: {model_config_path}")
    if not os.path.exists(save_scaler_dir_path):
        raise FileNotFoundError(f"Scaler directory path does not exist: {save_scaler_dir_path}")

    main(
        model_config_path=model_config_path,
        save_scaler_dir_path=save_scaler_dir_path,
        overwrite_preprocessed=overwrite_preprocessed,
        log_in_neptune=LOG_IN_NEPTUNE,
        neptune_run_name=neptune_run_name,
        neptune_tags=neptune_tags,
    )
