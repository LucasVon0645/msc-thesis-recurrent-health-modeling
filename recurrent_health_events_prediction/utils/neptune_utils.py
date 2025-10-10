from typing import Optional
import os
import neptune

from recurrent_health_events_prediction.utils.general_utils import import_yaml_config

def export_neptune_token_from_file(token_file_path: str):
    """
    Reads the Neptune API token from a file and sets it as an environment variable.

    Args:
        token_file_path (str): Path to the file containing the Neptune API token.

    Returns:
        None
    """
    if not os.path.exists(token_file_path):
        raise FileNotFoundError(f"Token file not found at {token_file_path}")

    with open(token_file_path, 'r') as file:
        os.environ["NEPTUNE_API_TOKEN"] = file.read().strip()

def add_plot_to_neptune_run(neptune_run: neptune.Run, filename: str, figure, path: Optional[str] ="plots"):
    """
    Adds a plot to a Neptune run.

    Args:
        neptune_run: The Neptune run object.
        filename (str): The name of the file to save the plot.
        figure: The figure object to log.

    Returns:
        None
    """
    filename_without_extension = os.path.splitext(filename)[0]
    if neptune_run is not None:
        neptune_run[f"{path}/{filename_without_extension}"].upload(neptune.types.File.as_image(figure))
        print(f"✅ Plot {filename_without_extension} added to Neptune run.")
    else:
        print("No Neptune run available, plot not added.")

def add_plotly_plots_to_neptune_run(neptune_run: neptune.Run, plotly_fig, filename: str, filepath: Optional[str] = "plots"):
    """
    Adds a Plotly figure to a Neptune run.s
    
    :param neptune_run: The Neptune run object.
    :param fig: The Plotly figure to log.
    :param filename: The name of the file to save the figure as.
    """
    if filename.endswith('.png') or filename.endswith('.jpg'):
        filename = filename[:-4]
        filename += '.html'
    elif not "."  in filename:
        filename += ".html"

    # Log the HTML file to Neptune
    neptune_run[f"{filepath}/{filename}"].upload(plotly_fig)

    print("✅ Plotly figure logged to Neptune run: ", filename)

def upload_model_to_neptune(neptune_run: neptune.Run, model_file_path: str, base_neptune_path: Optional[str] = "artifacts/models"):
    """
    Uploads the model file to Neptune under the same filename as on disk.

    Parameters:
    - neptune_run: Neptune run object
    - model_file_path: Path to the saved model file
    - base_neptune_path: Base path in Neptune (folder-like)
    """
    if os.path.exists(model_file_path):
        filename = os.path.basename(model_file_path)
        neptune_path = f"{base_neptune_path}/{filename}"
        neptune_run[neptune_path].upload(model_file_path)
        print(f"✅ Uploaded to Neptune: {neptune_path}")
    else:
        print(f"⚠️ Model file not found: {model_file_path}")

def upload_hmm_output_file_to_neptune(neptune_run: neptune.Run, output_file_path: str, base_neptune_path: Optional[str] = "artifacts/hmm_output"):
    """
    Uploads the HMM output file to Neptune under the same filename as on disk.

    Parameters:
    - neptune_run: Neptune run object
    - output_file_path: Path to the HMM output file
    - base_neptune_path: Base path in Neptune (folder-like)
    """
    if os.path.exists(output_file_path):
        filename = os.path.basename(output_file_path)
        neptune_path = f"{base_neptune_path}/{filename}"
        neptune_run[neptune_path].upload(output_file_path)
        print(f"✅ Uploaded to Neptune: {neptune_path}")
    else:
        print(f"⚠️ HMM output file not found: {output_file_path}")

def upload_training_data_to_neptune(neptune_run: neptune.Run, training_data_path: str, base_neptune_path: Optional[str] = "artifacts/training_data"):
    """
    Uploads the training data file to Neptune under the same filename as on disk.

    Parameters:
    - neptune_run: Neptune run object
    - training_data_path: Path to the training data file
    - base_neptune_path: Base path in Neptune (folder-like)
    """
    if os.path.exists(training_data_path):
        filename = os.path.basename(training_data_path)
        neptune_path = f"{base_neptune_path}/{filename}"
        neptune_run[neptune_path].upload(training_data_path)
        print(f"✅ Uploaded to Neptune: {neptune_path}")
    else:
        print(f"⚠️ Training data file not found: {training_data_path}")

def upload_file_to_neptune(
    neptune_run: neptune.Run,
    local_path: str,
    neptune_base_path: str = "artifacts",
    neptune_filename: Optional[str] = None,
) -> str:
    """
    Upload a local file to a Neptune run under a chosen path.

    Parameters
    ----------
    neptune_run : neptune.Run
        The active Neptune run object.
    local_path : str
        Path to the local file to upload.
    neptune_base_path : str, optional
        Folder-like path in Neptune under which the file will appear.
        Defaults to "artifacts".
    neptune_filename : str, optional
        Filename to use in Neptune. If None, uses the local filename.

    Returns
    -------
    str
        The full Neptune path the file was uploaded to.

    Raises
    ------
    FileNotFoundError
        If `local_path` does not exist.
    IsADirectoryError
        If `local_path` is a directory (this helper is for single files).
    """
    if not os.path.exists(local_path):
        print(f"⚠️ File not found: {local_path}")
        return None
    if not os.path.isfile(local_path):
        print(f"⚠️ `local_path` must be a file, not a directory: {local_path} (use upload_files / a different helper for folders).")
        return None

    base = neptune_base_path.strip().strip("/")  # normalize
    filename = neptune_filename or os.path.basename(local_path)
    neptune_path = f"{base}/{filename}" if base else filename

    neptune_run[neptune_path].upload(local_path)
    print(f"✅ Uploaded to Neptune: {neptune_path}")
    return neptune_path

def add_model_config_to_neptune(neptune_run: neptune.Run, model_config: dict):
    for key, value in model_config.items():
        if isinstance(value, list):
            try:
                # Convert list to "[item1, item2, ...]" string format
                neptune_run[f"model_config/{key}"] = "[" + ", ".join(map(str, value)) + "]"
            except Exception as e:
                print(f"Could not log list for key {key}: {e}")
        else:
            try:
                neptune_run[f"model_config/{key}"] = value
            except Exception as e:
                print(f"Could not log value for key {key}: {e}")

def stringify_dict_values_for_neptune(dictionary: dict, use_parenthesis = True) -> str:
    """
    Converts a dictionary to a string representation suitable for logging in Neptune.
    
    Args:
        dictionary (dict): The dictionary to convert.
        
    Returns:
        str: String representation of the dictionary.
    """
    if use_parenthesis:
        return '\n'.join(f"{key} ({value})" for key, value in dictionary.items())
    else:
        return '\n'.join(f"{key}: {value}" for key, value in dictionary.items())


def initialize_neptune_run(data_config_path: str, run_name: str, dataset, tags: Optional[list] = None) -> neptune.Run:
    data_config = import_yaml_config(data_config_path)

    token_file_path = data_config["neptune"]["token_file_path"]
    neptune_project = data_config["neptune"]["project"]
    export_neptune_token_from_file(token_file_path)

    tags = tags or []
    if dataset not in tags:
        tags.append(dataset)

    # Initialize neptune run
    run = neptune.init_run(
        project=neptune_project,
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
        name=run_name,
        tags=tags,
    )

    return run

def track_file_in_neptune(neptune_run, neptune_path: str, local_path: str):
    """
    Track a file or folder in Neptune.
    
    Args:
        neptune_run: The active Neptune run object.
        neptune_path (str): The path inside the Neptune UI (e.g. "training_data/files").
        local_path (str): The local path to the file or folder (absolute or relative).
    """
    neptune_run[neptune_path].track_files(local_path)
