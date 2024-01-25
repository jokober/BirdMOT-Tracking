import configparser
import os
from pathlib import Path
from typing import List

# Get paths of config folders
def get_local_data_path() -> Path:
    assert os.environ.get('SAHI_TRACKING_DATA_PATH') is not None, "SAHI_TRACKING_DATA_PATH environment variable not set. Please select a target path, where your data will be stored."
    return Path(os.environ.get("SAHI_TRACKING_DATA_PATH"))

def get_experiments_path() -> Path:
    return get_local_data_path() / 'config' / 'tracking_experiments'

# Get lists of available config files

def get_list_of_sahi_models() -> List[str]:
    return [path.name for path in (get_local_data_path() / 'config' / 'sahi_models').glob('*.pt')]

def get_list_of_sahi_prediction_params() -> List[str]:
    return [path.name for path in (get_local_data_path() / 'config' / 'sahi_prediction_params').glob('*.json')]

def get_list_of_tracking_datasets() -> List[str]:
    return [path.name for path in (get_local_data_path() / 'config' / 'tracking_datasets').glob('*.json')]

def get_list_of_experiments() -> List[str]:
    return [exp.name for exp in get_experiments_path().glob('*.json')]

# Get individual config files by name

def get_sahi_model_path_by_name(model_name: str) -> Path:
    return get_local_data_path() / 'config' / 'sahi_models' / model_name

def get_sahi_prediction_params_path_by_name(prediction_params_name: str) -> Path:
    return get_local_data_path() / 'config' / 'sahi_prediction_params' / prediction_params_name

def get_tracking_dataset_path_by_name(dataset_name: str) -> Path:
    return get_local_data_path() / 'config' / 'tracking_datasets' / dataset_name

def get_tracking_experiment_path_by_name(experiment_name: str) -> Path:
    return get_experiments_path() / experiment_name

# Get dataset paths

def get_coco_files_path() -> Path:
    return get_local_data_path() / 'dataset' / 'cocovid' / 'coco_files'

def get_datasets_path() -> Path:
    datasets_path =  get_local_data_path() / 'work_dir' / 'datasets'
    datasets_path.mkdir(exist_ok=True, parents=True)
    return datasets_path

# Get paths working directories

def get_predictions_path() -> Path:
    predictions_path = get_local_data_path() / 'work_dir' / 'predictions'
    predictions_path.mkdir(exist_ok=True, parents=True)
    return predictions_path

def get_tracking_results_path() -> Path:
    tracking_results_path =  get_local_data_path() / 'work_dir' / 'tracking_results'
    tracking_results_path.mkdir(exist_ok=True,  parents=True)
    return tracking_results_path

def get_evaluation_results_path() -> Path:
    eval_results_path =  get_local_data_path() / 'work_dir' / 'eval_results'
    eval_results_path.mkdir(exist_ok=True,  parents=True)
    return eval_results_path

