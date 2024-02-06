import argparse
import json
from argparse import ArgumentParser
from pathlib import Path

from sahi_tracking.experiments_framework.DataStatePersistance import DataStatePersistance
from sahi_tracking.experiments_framework.dataset_creation import find_or_create_dataset
from sahi_tracking.experiments_framework.evaluation_result_creation import find_or_create_tracker_evaluations
from sahi_tracking.experiments_framework.predictions_creation import find_or_create_predictions
from sahi_tracking.experiments_framework.tracking_result_creation import find_or_create_tracking_results
from sahi_tracking.experiments_framework.utils import load_config_files
from sahi_tracking.helper.config import get_local_data_path


def run_experiment_framework(tracking_dataset_dict, detection_params_dict, tracking_experiment_dict,
                             model_path=None, device='cpu', overwrite_existing=False, read_only=False):
    persistence_state = DataStatePersistance(read_only)

    dataset = find_or_create_dataset(tracking_dataset_dict,
                                     persistence_state=persistence_state,
                                     cocovid_img_path=get_local_data_path() / "dataset/cocovid/images",
                                     overwrite_existing=overwrite_existing)

    predictions_result = find_or_create_predictions(dataset,
                                                    prediction_params=detection_params_dict,
                                                    model_path=model_path,
                                                    persistence_state=persistence_state,
                                                    device=device,
                                                    cocovid_img_path=get_local_data_path() / "dataset/cocovid/images",
                                                    overwrite_existing=overwrite_existing)

    for one_experiment in tracking_experiment_dict['tracker_experiments']:
        tracking_results = find_or_create_tracking_results(tracking_experiment=one_experiment,
                                                           predictions_result=predictions_result,
                                                           dataset=dataset,
                                                           persistence_state=persistence_state,
                                                           overwrite_existing=overwrite_existing)

        evaluation_results = find_or_create_tracker_evaluations(tracking_results=tracking_results,
                                                                predictions_result=predictions_result,
                                                                dataset=dataset,
                                                                persistence_state=persistence_state,
                                                                overwrite_existing=overwrite_existing)

    return evaluation_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tracking_dataset_path", type=Path, required=True)
    parser.add_argument("--tracking_experiment_path", type=Path, required=True)
    parser.add_argument("--detection_params_path", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=False)
    parser.add_argument("--caching", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--overwrite_existing", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--device", type=str, required=False, default='cpu')
    args = parser.parse_args()

    tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict = load_config_files(
        args.tracking_dataset_path, args.tracking_experiment_path, args.detection_params_path
    )

    run_experiment_framework(tracking_dataset_dict, sahi_predictions_params_dict, tracking_experiment_dict,
                             args.model_path, args.device, args.overwrite_existing)
