import os
from importlib import resources as impresources
from importlib import import_module
import yaml
from typing import Optional, Tuple

import pandas as pd
import torch
import torchmetrics

from recurrent_health_events_prediction import configs
from recurrent_health_events_prediction.datasets.HospReadmDataset import HospReadmDataset
from recurrent_health_events_prediction.preprocessing.utils import remap_discharge_location, remap_mimic_races
from recurrent_health_events_prediction.training.utils import preprocess_features_to_one_hot_encode, standard_scale_data
from recurrent_health_events_prediction.utils.general_utils import import_yaml_config

def create_preprocessed_data(
    data_directory: str,
    training_data_config: dict,
    save_scaler_dir_path: str = None
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
    features_to_scale = training_data_config['features_to_scale']
    features_to_one_hot_encode = training_data_config['features_to_one_hot_encode']
    one_hot_cols_to_drop = training_data_config['one_hot_cols_to_drop']

    train_scaled_df, test_scaled_df = standard_scale_data(
        train_df, test_df, features_to_scale, save_scaler_dir_path=save_scaler_dir_path
    )

    train_preprocessed_df, _ = preprocess_features_to_one_hot_encode(
        train_scaled_df, features_to_encode=features_to_one_hot_encode,
        one_hot_cols_to_drop=one_hot_cols_to_drop
    )

    test_preprocessed_df, _ = preprocess_features_to_one_hot_encode(
        test_scaled_df, features_to_encode=features_to_one_hot_encode,
        one_hot_cols_to_drop=one_hot_cols_to_drop
    )
    
    train_preprocessed_file_path = os.path.join(data_directory, "train_events_preprocessed.csv")
    test_preprocessed_file_path = os.path.join(data_directory, "test_events_preprocessed.csv")

    train_preprocessed_df.to_csv(train_preprocessed_file_path, index=False)
    test_preprocessed_df.to_csv(test_preprocessed_file_path, index=False)

    return train_preprocessed_file_path, test_preprocessed_file_path

def get_train_test_datasets(train_df_path, test_df_path, model_config, training_data_config):
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
        'longitudinal_feat_cols': model_config['longitudinal_feat_cols'],
        'current_feat_cols': model_config['current_feat_cols'],
        'max_seq_len': model_config.get('max_sequence_length', 5),
        'no_elective': model_config.get('no_elective', True),
        'reverse_chronological_order': model_config.get('reverse_chronological_order', True),
        # column names config
        'subject_id_col': training_data_config.get('patient_id_col', "SUBJECT_ID"),
        'order_col': training_data_config.get('time_col', "ADMITTIME"),
        'label_col': training_data_config.get('binary_event_col', "READMISSION_30_DAYS"),
        'next_admt_type_col': model_config.get("next_admt_type_col", "NEXT_ADMISSION_TYPE"),
        'hosp_id_col': model_config.get("hosp_id_col", "HADM_ID"),
    }

    # Create datasets
    train_dataset = HospReadmDataset(
        csv_path=train_df_path,
        **dataset_config
    )

    test_dataset = HospReadmDataset(
        csv_path=test_df_path,
        **dataset_config
    )

    return train_dataset, test_dataset

def train(
    train_dataset: HospReadmDataset,
    model_config: dict,
    ModelClass: Optional[torch.nn.Module] = None) -> Tuple[torch.nn.Module, list[float]]:
    """
    Train the model.

    Args:
        train_dataset (HospReadmDataset): Training dataset.
        model_config (dict): Model configuration parameters.
    Returns:
        Trained model and training loss history.
    """
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True)
    
    # Initialize model
    if ModelClass is not None:
        print(f"\nUsing provided model class: {ModelClass.__name__}")
    else:
        model_class_name = model_config.get('model_class', 'GRUNet')

        # Class resolver: assume `model_class_name` is a class defined in
        # `recurrent_health_events_prediction.model.RecurrentHealthEventsDL`.
        mod = import_module("recurrent_health_events_prediction.model.RecurrentHealthEventsDL")
        try:
            ModelClass = getattr(mod, model_class_name)
            print(f"\nUsing model class: {model_class_name}")
        except AttributeError:
            raise ImportError(f"Model class '{model_class_name}' not found in RecurrentHealthEventsDL")

    model: torch.nn.Module = ModelClass(**model_config['model_params'])

    print("\nModel initialized and ready for training.")
    print("Model parameters:")
    for key, value in model_config['model_params'].items():
        print(f"  {key}: {value}")

    # Set up loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])

    model.train()
    loss_over_epochs = []
    
    print("\nStarting training...")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Batch size: {model_config['batch_size']}")
    print(f"Learning rate: {model_config['learning_rate']}")
    print("Optimizer: Adam")
    print("Loss function: CrossEntropyLoss\n")
    
    # Training loop
    for epoch in range(model_config['num_epochs']):
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
        loss_over_epochs.append(epoch_loss / len(train_dataloader))
        print(f"Epoch {epoch + 1}/{model_config['num_epochs']}, Loss: {epoch_loss / len(train_dataloader)}")

    print("\nTraining complete.")
    
    return model, loss_over_epochs

def evaluate(test_dataset: HospReadmDataset, model: torch.nn.Module, batch_size: 64, prob_threshold: float = 0.5) -> dict:
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
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "auroc": auroc
    }

def main():
    model_name = "gru"
    save_scaler_dir_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/scalers"
    model_config_path = f"/workspaces/msc-thesis-recurrent-health-modeling/_models/mimic/deep_learning/{model_name}/{model_name}_config.yaml"
    overwrite_preprocessed = False

    with open((impresources.files(configs) / 'data_config.yaml')) as f:
        data_config = yaml.safe_load(f)

    training_data_config = data_config['training_data']["mimic"]
    data_directory = training_data_config['data_directory']

    model_config = import_yaml_config(model_config_path)

    # Load and preprocess data
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

    train_dataset, test_dataset = get_train_test_datasets(
        train_df_path, test_df_path,
        model_config,
        training_data_config=training_data_config
    )

    model, loss_over_epochs = train(
        train_dataset,
        model_config
    )

    eval_results = evaluate(test_dataset, model, batch_size=model_config['batch_size'])

    print("Training and evaluation complete.")