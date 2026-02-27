import json

def load_config(config_path):
    """
    Load configuration parameters from a JSON file.
    Args:
        config_path (str): The path to the configuration JSON file.
    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config