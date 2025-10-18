import os
from typing import Optional, Tuple

import neptune
import pandas as pd
import torch

from recurrent_health_events_prediction.datasets.HospReadmDataset import (
    HospReadmDataset,
)
from recurrent_health_events_prediction.model.utils import plot_auc
from recurrent_health_events_prediction.training.utils import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pred_proba_distribution,
    standard_scale_data,
)
from recurrent_health_events_prediction.utils.general_utils import add_suffix_before_ext
from recurrent_health_events_prediction.utils.neptune_utils import (
    add_plot_to_neptune_run,
    add_plotly_plots_to_neptune_run,
)

def scale_preprocessed_data(
    train_file_path: str,
    test_file_path: str,
    training_data_config: dict,
    save_scaler_dir_path: str | None = None,
    output_dir: str | None = None,
    output_train_filename: str = "train_events_preprocessed.csv",
    output_test_filename: str = "test_events_preprocessed.csv",
) -> Tuple[str, str]:
    """
    Scale data for training and testing/validation using features listed in
    `training_data_config['features_to_scale']`. Fits scaler on train and
    applies to test/val. Saves two CSVs and returns their paths.
    """
    # --- Load ---
    train_df = pd.read_csv(train_file_path)
    test_df  = pd.read_csv(test_file_path)

    # --- Validate config & columns ---
    features_to_scale = training_data_config.get("features_to_scale", None)
    if not features_to_scale:
        raise ValueError("`features_to_scale` missing or empty in training_data_config.")

    missing_train = [c for c in features_to_scale if c not in train_df.columns]
    missing_test  = [c for c in features_to_scale if c not in test_df.columns]
    if missing_train:
        raise KeyError(f"Columns missing in TRAIN for scaling: {missing_train}")
    if missing_test:
        raise KeyError(f"Columns missing in TEST/VAL for scaling: {missing_test}")

    # --- Scale (fit on train, transform both) ---
    train_scaled_df, test_scaled_df = standard_scale_data(
        train_df, test_df, features_to_scale, save_scaler_dir_path=save_scaler_dir_path
    )

    # --- Write outputs ---
    data_directory = output_dir if output_dir is not None else os.path.dirname(train_file_path)
    os.makedirs(data_directory, exist_ok=True)

    train_out = os.path.join(data_directory, output_train_filename)
    test_out  = os.path.join(data_directory, output_test_filename)

    train_scaled_df.to_csv(train_out, index=False)
    test_scaled_df.to_csv(test_out, index=False)

    return train_out, test_out

def preprocess_pair(
    *,
    raw_train_csv: str,
    raw_eval_csv: str,
    training_data_config: dict,
    save_scaler_dir_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """
    Given raw CSVs, write preprocessed CSVs named '<raw>_preprocessed.csv' in output_dir (or raw dir).
    Returns paths to (train_preproc_csv, eval_preproc_csv).
    """
    # Decide output filenames by suffix rule
    train_out = add_suffix_before_ext(
        os.path.join(output_dir or os.path.dirname(raw_train_csv),
                     os.path.basename(raw_train_csv)),
        "_preprocessed"
    )
    eval_out  = add_suffix_before_ext(
        os.path.join(output_dir or os.path.dirname(raw_eval_csv),
                     os.path.basename(raw_eval_csv)),
        "_preprocessed"
    )

    # Skip if already there and not overwriting
    if (not overwrite) and os.path.exists(train_out) and os.path.exists(eval_out):
        print("Preprocessed files already exist and overwrite is not enabled.")
        print(f"Train preprocessed CSV: {train_out}")
        print(f"Eval preprocessed CSV: {eval_out}")
        return train_out, eval_out

    # Ensure output directory exists
    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    print("Preprocessing and scaling data...")
    
    # Reuse your existing function (the improved one you wrote)
    # – fits on train, transforms eval, saves CSVs.
    _train_out, _eval_out = scale_preprocessed_data(
        train_file_path=raw_train_csv,
        test_file_path=raw_eval_csv,
        training_data_config=training_data_config,
        save_scaler_dir_path=save_scaler_dir_path,
        output_dir=os.path.dirname(train_out),
        output_train_filename=os.path.basename(train_out),
        output_test_filename=os.path.basename(eval_out),
    )
    
    print(f"Preprocessed train CSV saved to: {_train_out}")
    print(f"Preprocessed eval CSV saved to: {_eval_out}")

    return _train_out, _eval_out

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

def build_or_load_datasets(
    *,
    preproc_train_csv: str,
    preproc_eval_csv: str,
    model_config: dict,
    training_data_config: dict,
    cache_dir: str,
    train_pt_name: str,
    eval_pt_name: str,
    compute_last_events_eval: bool = False,
    last_events_pt_name: Optional[str] = None,
    overwrite_pt: bool = False,
) -> Tuple["HospReadmDataset", "HospReadmDataset", Optional["HospReadmDataset"]]:
    """
    Create/load (train, eval) datasets from preprocessed CSVs; optionally last-events eval dataset.
    Saves/loads from cache_dir with given pt names.
    """
    os.makedirs(cache_dir, exist_ok=True)
    train_pt = os.path.join(cache_dir, train_pt_name)
    eval_pt  = os.path.join(cache_dir, eval_pt_name)
    last_pt  = os.path.join(cache_dir, last_events_pt_name) if (compute_last_events_eval and last_events_pt_name) else None

    can_load = (not overwrite_pt) and os.path.exists(train_pt) and os.path.exists(eval_pt) \
               and ((not compute_last_events_eval) or (last_pt and os.path.exists(last_pt)))

    if can_load:
        print("Loading datasets from cached .pt files...")
        train_ds = torch.load(train_pt, weights_only=False)
        eval_ds  = torch.load(eval_pt,  weights_only=False)
        last_ds  = torch.load(last_pt,  weights_only=False) if (compute_last_events_eval and last_pt) else None
        return train_ds, eval_ds, last_ds

    # Build fresh datasets
    print("Building datasets from preprocessed CSV files...")
    train_ds, eval_ds = get_train_test_datasets(
        preproc_train_csv,
        preproc_eval_csv,
        model_config,
        training_data_config=training_data_config,
    )  # uses your dataset builder:contentReference[oaicite:3]{index=3}

    last_ds = None
    if compute_last_events_eval:
        last_ds = get_test_last_events_only_dataset(
            preproc_eval_csv,
            model_config,
            training_data_config=training_data_config,
        )  # builds last-events dataset when you need it:contentReference[oaicite:4]{index=4}

    # Cache
    torch.save(train_ds, train_pt)
    torch.save(eval_ds,  eval_pt)
    if compute_last_events_eval and last_pt and last_ds is not None:
        torch.save(last_ds, last_pt)

    return train_ds, eval_ds, last_ds

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

def plot_evaluation_figures(
    all_labels,
    all_pred_probs,
    model_name,
    log_dir,
    class_names_dict=None,
    neptune_run=None,
):
    """
    Plot evaluation figures: predicted probability distribution, ROC curve, and calibration curve.
    """
    # --- Plot evaluation figures ---
    fig_hist = plot_pred_proba_distribution(
        all_labels, all_pred_probs, show_plot=False, class_names=class_names_dict
    )
    fig_hist = fig_hist.update_layout(
        title=f"Predicted Probabilities by True Labels - {model_name}"
    )
    fig_hist.write_html(os.path.join(log_dir, "pred_proba_distribution.html"))

    fig_auc = plot_auc(
        all_pred_probs,
        all_labels,
        show_plot=False,
        title=f"ROC Curve - {model_name}",
        save_path=os.path.join(log_dir, "roc_curve.html"),
    )

    fig_cal = plot_calibration_curve(
        all_labels,
        all_pred_probs,
        show_plot=False,
        title=f"Calibration Curve - {model_name}",
        save_path=os.path.join(log_dir, "calibration_curve.html"),
    )

    # --- Log to Neptune (if applicable) ---
    if neptune_run:
        neptune_path = "evaluation/last_events"

        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_hist,
            filename="pred_proba_distribution.html",
            filepath=neptune_path,
        )
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_auc,
            filename="roc_curve.html",
            filepath=neptune_path,
        )
        add_plotly_plots_to_neptune_run(
            neptune_run,
            fig_cal,
            filename="calibration_curve.html",
            filepath=neptune_path,
        )

    return {
        "hist": fig_hist,
        "roc": fig_auc,
        "calibration": fig_cal,
    }

def make_tb_writer(log_dir: str | None = None):
    # e.g., logs go under runs/gru_model_run/<timestamp>
    from torch.utils.tensorboard import SummaryWriter
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)