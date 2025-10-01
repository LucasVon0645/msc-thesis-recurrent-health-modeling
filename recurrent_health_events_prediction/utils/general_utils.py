import yaml
import os

def import_yaml_config(config_path: str) -> dict:
    """
    Imports a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_yaml_config(config: dict, config_path: str) -> None:
    """
    Saves a dictionary as a YAML configuration file.

    Args:
        config (dict): Configuration to save.
        config_path (str): Path where the configuration will be saved.
    """
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    
def check_if_directory_exists(directory_path: str) -> bool:
    """
    Checks if a directory exists.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        bool: True if the directory exists, False otherwise.
    """
    return os.path.isdir(directory_path)

def check_if_file_exists(file_path: str) -> bool:
    """
    Checks if a file exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def stringify_dict_values(dictionary: dict, use_parenthesis = True) -> str:
    """
    Converts a dictionary to a string representation.
    
    Args:
        dictionary (dict): The dictionary to convert.
        
    Returns:
        str: String representation of the dictionary.
    """
    if use_parenthesis:
        return '\n'.join(f"{key} ({value})" for key, value in dictionary.items())
    else:
        return '\n'.join(f"{key}: {value}" for key, value in dictionary.items())