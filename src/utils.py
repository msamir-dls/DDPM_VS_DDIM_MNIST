import os
import yaml
import time
import torch
import mlflow
import functools
from pathlib import Path

def time_execution(func):
    """
    Decorator that calculates execution time and logs it to MLflow.
    Useful for comparing DDPM vs DDIM inference speeds.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        # Log to console
        print(f"[*] Method '{func.__name__}' executed in {duration:.4f} seconds.")
        
        # Log to MLflow if an active run exists
        if mlflow.active_run():
            # Distinguish between training and inference timing
            metric_name = f"time_{func.__name__}"
            mlflow.log_metric(metric_name, duration)
            
        return result
    return wrapper

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow(config):
    """Initializes MLflow experiment and logs hyperparameters."""
    mlflow.set_tracking_uri(config.get('mlflow', {}).get('tracking_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Flatten the config dictionary to log as parameters
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    mlflow.log_params(flatten_dict(config))

def save_checkpoint(model, optimizer, epoch, path):
    """Saves model weights professionally."""
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"[+] Checkpoint saved at {path}")

def get_device():
    """Returns the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")