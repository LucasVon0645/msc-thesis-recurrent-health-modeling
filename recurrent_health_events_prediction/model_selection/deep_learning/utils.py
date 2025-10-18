import copy
import os
import optuna
from pathlib import Path
from optuna.importance import get_param_importances
import yaml

from recurrent_health_events_prediction.utils.neptune_utils import upload_file_to_neptune

def save_space_to_txt(space_hyperparams, out_dir):
    """
    Save your hyperparameter search space (dict of lists/tuples)
    to a plain text file.
    """
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"hyperparams_search_space.txt")

    with open(filepath, "w") as f:
        f.write("### Hyperparameter Search Space ###\n\n")
        for name, values in space_hyperparams.items():
            # pretty formatting for tuples/lists/numbers
            if isinstance(values, (list, tuple)):
                f.write(f"{name}: {list(values)}\n")
            else:
                f.write(f"{name}: {values}\n")

    print(f"Search space saved to: {filepath}")
    return filepath


def save_study_artifacts(
    study: optuna.study.Study,
    out_dir: str,
    base_config: dict,
    model_class_name: str,
    neptune_run=None
):
    """
    Writes:
      - best_params.yaml
      - trials.csv
      - param_importances.yaml
      - (optional) best_config.yaml and best_model.pth after recomputing metrics
    Also prints best metrics captured during HPO (from trial.user_attrs).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- a) Best params → YAML
    best_params = study.best_trial.params
    best_params_path = out / "best_params.yaml"
    print(f"Best params saved to: {best_params_path}")
    with open(best_params_path, "w") as f:
        yaml.safe_dump(best_params, f, sort_keys=True)
    if neptune_run:
        neptune_run["best_params"] = best_params

    # --- b) Trials → CSV
    df = study.trials_dataframe(attrs=(
        "number", "value", "params", "state", "datetime_start", "datetime_complete"
    ))
    trials_path = out / "trials.csv"
    print(f"Trials saved to: {trials_path}")
    df.to_csv(trials_path, index=False)
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=str(trials_path),
            neptune_base_path="artifacts",
            neptune_filename="trials.csv"
        )

    # --- c) Parameter importances → YAML (may fail if too few trials)
    try:
        imps = get_param_importances(study)
        if neptune_run:
            neptune_run["param_importances"] = imps
        with open(out / "param_importances.yaml", "w") as f:
            yaml.safe_dump({k: float(v) for k, v in imps.items()}, f, sort_keys=False)
    except Exception as e:
        print(f"[warn] Could not compute importances: {e}")

    # --- d) Best metrics as recorded during tuning (no retrain)
    bt = study.best_trial
    best_auroc = bt.user_attrs.get("auroc")
    best_f1    = bt.user_attrs.get("f1")
    print(f"Best trial number: {bt.number}")
    print(f"Best recorded metrics (during HPO): AUROC={best_auroc}, F1={best_f1}")
    
    if neptune_run:
        neptune_run["best_trial_number"] = bt.number
        neptune_run["best_auroc"] = best_auroc
        neptune_run["best_f1"] = best_f1
    
    # Build best config for reference
    best_config = copy.deepcopy(base_config)

    # Inject best hyperparams back into config
    # (must match how you injected inside objective())
    p = best_params
    best_config["learning_rate"] = p["learning_rate"]
    best_config["batch_size"]    = p["batch_size"]
    best_config["model_params"]["hidden_size_head"] = p["hidden_size_head"]
    best_config["model_params"]["dropout"]          = p["dropout"]

    if model_class_name == "GRUNet":
        best_config["model_params"]["hidden_size_seq"] = p["hidden_size_seq"]
        best_config["model_params"]["num_layers_seq"]  = p["num_layers_seq"]
        best_config["model_class"] = "GRUNet"
    else:
        best_config["model_params"]["hidden_size_seq"] = p["hidden_size_seq"]
        best_config["model_class"] = "AttentionPoolingNet"
    
    best_config_path = out / "best_config.yaml"
    with open(best_config_path, "w") as f:
        yaml.safe_dump(best_config, f, sort_keys=False)
    print(f"Best config saved to: {best_config_path}")
    if neptune_run:
        upload_file_to_neptune(
            neptune_run,
            local_path=str(best_config_path),
            neptune_base_path="artifacts",
            neptune_filename="best_config.yaml"
        )
