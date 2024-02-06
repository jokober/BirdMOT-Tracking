import json
from pathlib import Path
from typing import Tuple


def load_config_files(tracking_dataset_path: Path, tracking_experiment_path: Path,
                      sahi_predictions_params_path: Path) -> Tuple[dict, dict, dict]:
    tracking_dataset_path = tracking_dataset_path.resolve()
    assert tracking_dataset_path, f"Tracking dataset {tracking_dataset_path} does not exist"
    with open(tracking_dataset_path) as json_file:
        tracking_dataset_dict = json.load(json_file)

    tracking_experiment_path = tracking_experiment_path.resolve()
    assert tracking_experiment_path, f"Experiment config {tracking_experiment_path} does not exist"
    with open(tracking_experiment_path) as json_file:
        tracking_experiment_dict = json.load(json_file)

    sahi_predictions_params_path = sahi_predictions_params_path.resolve()
    assert sahi_predictions_params_path, f"Sahi predictions params {sahi_predictions_params_path} does not exist"
    with open(sahi_predictions_params_path) as json_file:
        sahi_predictions_params_dict = json.load(json_file)

    return tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict
